import string

import numpy

from chainer import cuda, function_node
from chainer.utils import type_check


__all__ = ['group_normalization']


# References for reduction algorithm.
# https://github.com/cupy/cupy/blob/master/cupy/core/reduction.pxi
# http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf


# Idea of better kernels.
#
# 1. Structuring both of forward and backward reductions into two kernels:
# HW reduction and group-batch reduction. The first kernel consists of
# block-strided loop and tree-sum, just as the current forward kernel does.
# The second kernel works on group-batch panels. The x-dimension of a block is
# the size of a group, and the y-dimension covers batch elements as much as
# possible. Thus the number of blocks is same as the number of groups.
# Seemingly this strategy can almost fully utilize parallelism while keeping
# implementation relatively simple. This strategy is especially efficient
# when batch size is small.
#
# 2. AtomicAdd: atomic operations are rather fast since Kepler gen.
# Using those greatly simplifies implementation while achieving almost
# optimal performance. The obvious downside is inherent nondeterministy.


# Possible optimizations of current implementation.
#
# 1. Hierarchical kernel invocations: currently reductions over values that
# doesn't fit in a block is done via a loop in the kernel. It can limit
# parallelism on GPU, given that ``batch_size * n_group`` would be smaller than
# the number of Streaming Multiprocessor. Multiple step reduction in
# the hierarchical manner solves the problem.
#
# 2. More use of vectorized values: using ``float2``-valued arrays for data
# passed between kernels (e.g. ``mu`` and ``inv_std``) may enhance memory
# access performance. Downside is necessity of raw-typed arguments for
# ``cuda.elementeise``.
#
# 3. Disassembling ``float2`` values at intra-block reduction: ``add_to_left``
# function can be rewritten to work on ``float`` values instead of
# ``float2`` values. By this change, the number of threads utilized doubles.
# It could decrease performance due to increased working threads, so
# benchmarking or profiling should be made.
#
# 4. Using ``__shfl_xor``: since compute capability 3.0 (Kepler gen.),
# ``__shfl_xor`` can be used for efficient intra-warp communication instead of
# shared memory. Note that on CUDA 9.0 or later ``__shfl_xor_sync`` must be
# used.
#
# 5. Custom kernel for reduction over batch elements: ``cuda.reduce`` is
# used for reduction over batch elements, to compute the gradient
# ``beta`` and ``gamma``. On current implementation (CuPy 2.1 and 4.0 checked),
# memory access pattern isn't tailored for reduction over the outermost axis.
# But probably the performance gain by custom kernel is limited, because
# that reduction takes only little part of memory accesses at the gradient
# computation.


# The maximum block size is 1024 since compute capability 2.0 (Fermi gen.)
# https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
_REDUCE_BLOCK_SIZES = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024


_common_cuda_code = '''
#if __CUDACC_VER_MAJOR__ < 9
#define __syncwarp()
#define __any_sync(mask, pred) __any(pred)
#endif


#define tid (threadIdx.y * blockDim.x + threadIdx.x)


static
__device__ __inline__
void add_to_left(float2 *vl, float2 *shm, unsigned int stride, int cond) {
    if (cond) {
        float2 vr = shm[tid + stride];
        vl->x += vr.x;
        vl->y += vr.y;
        if (stride > 1) {
            if (stride <= 32) __syncwarp();
            shm[tid] = *vl;
            if (stride <= 32) __syncwarp();
        }
    }
    if (stride > 32) __syncthreads();
}
'''


_fwd_reduce_cuda_code = _common_cuda_code + '''
// Parametalized compilation.
#define FWD_BLOCK_SIZE ${fwd_block_size}


extern "C"
__global__
void gn_fwd_red_kernel(
        float *x, float eps, float *mu, float *inv_std,
        unsigned int n_elems_per_group, float inv_n_elems_per_group) {
    // For each group, one block is assigned.
    // Requires that FWD_BLOCK_SIZE <= n_elems_per_group.
#if FWD_BLOCK_SIZE >= 2
    __shared__ float2 shm_fwd_red[FWD_BLOCK_SIZE >= 64 ? FWD_BLOCK_SIZE : 64];
#else
    float2 *shm_fwd_red;  // Dummy for compilation.
#endif

    // Loop reduction.
    unsigned int group_begin = blockIdx.x * n_elems_per_group;
    unsigned int group_cur = group_begin + threadIdx.x;
    unsigned int group_end = group_begin + n_elems_per_group;

    float v = x[group_cur];
    float val_sum = v;
    float val_sqsum = v * v;

    if (FWD_BLOCK_SIZE >= 1) {
        while ((group_cur += FWD_BLOCK_SIZE) < group_end) {
            v = x[group_cur];
            val_sum += v;
            val_sqsum += v * v;
        }
    }

    // In-block reduction.
    float2 vl = make_float2(val_sum, val_sqsum);
    if (FWD_BLOCK_SIZE >= 2) {
        shm_fwd_red[tid] = vl;
    }

    if (FWD_BLOCK_SIZE >= 64) {
        __syncthreads();
    } else if (FWD_BLOCK_SIZE >= 2) {
        __syncwarp();
    }

#define add_to_left_fwd_red(stride) add_to_left( \
    &vl, shm_fwd_red, stride, tid < stride || stride <= 32)

    if (FWD_BLOCK_SIZE >= 1024) {
        add_to_left_fwd_red(512);
    }
    if (FWD_BLOCK_SIZE >= 512) {
        add_to_left_fwd_red(256);
    }
    if (FWD_BLOCK_SIZE >= 256) {
        add_to_left_fwd_red(128);
    }
    if (FWD_BLOCK_SIZE >= 128) {
        add_to_left_fwd_red(64);
    }
    if (tid < 32) {
        if (FWD_BLOCK_SIZE >= 64) {
            add_to_left_fwd_red(32);
        }
        if (FWD_BLOCK_SIZE >= 32) {
            add_to_left_fwd_red(16);
        }
        if (FWD_BLOCK_SIZE >= 16) {
            add_to_left_fwd_red(8);
        }
        if (FWD_BLOCK_SIZE >= 8) {
            add_to_left_fwd_red(4);
        }
        if (FWD_BLOCK_SIZE >= 4) {
            add_to_left_fwd_red(2);
        }
        if (FWD_BLOCK_SIZE >= 2) {
            add_to_left_fwd_red(1);
        }
    }

    // Post-map.
    if (tid == 0) {
        float val_mu = vl.x * inv_n_elems_per_group;
        mu[blockIdx.x] = val_mu;
        float var = vl.y * inv_n_elems_per_group - val_mu * val_mu;
        inv_std[blockIdx.x] = rsqrt(var + eps);
    }
}
'''


@cuda.memoize(for_each_device=True)
def _get_fwd_reduce_cuda_kernel(fwd_block_size):
    assert fwd_block_size in (0,) + _REDUCE_BLOCK_SIZES
    code = string.Template(_fwd_reduce_cuda_code).substitute(
        fwd_block_size=fwd_block_size)
    prog = cuda.cupy.cuda.compile_with_cache(code)
    return prog.get_function('gn_fwd_red_kernel')


def _call_fwd_reduce_cuda_kernel(x, eps, shape):
    batch_size, groups, n_channels_per_group, n_elems_per_channel = shape

    mu = cuda.cupy.empty((batch_size * groups), dtype=x.dtype)
    inv_std = cuda.cupy.empty((batch_size * groups), dtype=x.dtype)

    n_elems_per_group = n_channels_per_group * n_elems_per_channel
    fwd_block_size = 0
    for fwd_block_size_candidate in _REDUCE_BLOCK_SIZES:
        if fwd_block_size_candidate < n_elems_per_group:
            fwd_block_size = max(fwd_block_size, fwd_block_size_candidate)
    kern = _get_fwd_reduce_cuda_kernel(fwd_block_size)

    blocks = batch_size * groups, 1, 1
    threads = fwd_block_size, 1, 1
    kern(blocks, threads, args=(
         x, x.dtype.type(eps), mu, inv_std,
         numpy.uint32(n_elems_per_group), x.dtype.type(1 / n_elems_per_group)))

    return mu, inv_std


class GroupNormalization(function_node.FunctionNode):

    """Group normalization."""

    def __init__(self, groups=32, eps=1e-5):
        self.groups = groups
        self.eps = eps
        self._x_hat = None
        self._inv_std = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, gamma_type, beta_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            gamma_type.ndim == 1,
            beta_type.ndim == 1,
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            x_type.shape[1] % self.groups == 0,
            x_type.shape[1] == gamma_type.shape[0],
            gamma_type.shape == beta_type.shape,
        )
        # TODO: make kernel accepts float16 or float64 values.
        type_check.expect(x_type.dtype == numpy.float32)

    def forward_cpu(self, inputs):
        self.retain_inputs((1,))
        x, gamma, beta = inputs

        orig_shape = x.shape
        batch_size, n_channels = orig_shape[:2]
        groups, n_channels_per_group = \
            self.groups, n_channels // self.groups
        x = x.reshape((batch_size, groups, n_channels_per_group, -1))
        gamma = gamma.reshape((groups, n_channels_per_group, 1))
        beta = beta.reshape((groups, n_channels_per_group, 1))

        mu = x.mean(axis=(2, 3), keepdims=True)
        var = (numpy.square(x).mean(axis=(2, 3), keepdims=True)
               - numpy.square(mu))
        inv_std = (var + self.eps) ** -0.5

        x_hat = (x - mu) * inv_std
        y = x_hat * gamma + beta

        self._x_hat, self._inv_std = x_hat, inv_std
        return y.reshape(orig_shape),

    def forward_gpu(self, inputs):
        self.retain_inputs((1,))
        x, gamma, beta = inputs
        if x.size == 0:
            return x,

        orig_shape = x.shape
        batch_size, n_channels = orig_shape[:2]
        n_elems_per_channel = numpy.prod(orig_shape[2:])
        groups, n_channels_per_group = \
            self.groups, n_channels // self.groups
        shape = batch_size, groups, n_channels_per_group, n_elems_per_channel

        x = x.ravel()
        gamma = gamma.ravel()
        beta = beta.ravel()

        x = cuda.cupy.ascontiguousarray(x)
        mu, inv_std = _call_fwd_reduce_cuda_kernel(x, self.eps, shape)

        # Use raw indexing with specialized compilation to avoid overheads
        # caused by indexing calculation, which involves successive integer
        # divmod operations. The optimization gives massive performance gain.
        gn_fwd_norm_code = string.Template('''
            unsigned int u = i;
            unsigned int tmp = u / ${n_elems_per_channel};
            unsigned int channel_idx = tmp % ${n_channels_per_group};
            tmp /= ${n_channels_per_group};
            unsigned int group_idx = tmp % ${groups};
            unsigned int batch_idx = tmp / ${groups};
            unsigned int group_norm_idx =
                batch_idx * ${groups} +
                group_idx;
            unsigned int batch_norm_idx =
                group_idx * ${n_channels_per_group} +
                channel_idx;
            T v_x_hat = (x[u] - mu[group_norm_idx]) * inv_std[group_norm_idx];
            x_hat[u] = v_x_hat;
            y[u] = v_x_hat * gamma[batch_norm_idx] + beta[batch_norm_idx];
        ''').substitute(n_elems_per_channel=n_elems_per_channel,
                        n_channels_per_group=n_channels_per_group,
                        groups=groups)
        gn_fwd_norm_name = 'gn_fwd_norm_{}_{}_{}'.format(
                    n_elems_per_channel, n_channels_per_group, groups)
        gn_fwd_norm_kern = cuda.elementwise(
            'raw T x, raw T mu, raw T inv_std, raw T gamma, raw T beta',
            'raw T x_hat, raw T y',
            gn_fwd_norm_code, gn_fwd_norm_name)

        x_hat, y = gn_fwd_norm_kern(x, mu, inv_std, gamma, beta,
                                    size=x.size)

        self._x_hat, self._inv_std = x_hat, inv_std
        return y.reshape(orig_shape),

    def backward(self, indexes, grad_outputs):
        gamma, = self.get_retained_inputs()
        gy, = grad_outputs
        f = GroupNormalizationGrad(self.groups, self._x_hat, self._inv_std)
        return f.apply((gamma, gy))


_bwd_reduce_cuda_code = _common_cuda_code + '''
// Parametalized compilation.
#define BWD_BLK_X_SIZE ${bwd_blk_x_size}
#define BWD_BLK_Y_SIZE ${bwd_blk_y_size}


#define BWD_BLOCK_SIZE \
    ((BWD_BLK_X_SIZE > 1 ? BWD_BLK_X_SIZE : 1) * BWD_BLK_Y_SIZE)


static
__device__ __inline__
void gn_bwd_elems_red_kernel(
        float *gy, float *x_hat,
        float *gy_sum, float *gy_x_hat_sum,
        unsigned int map_cur, unsigned int map_end, int is_channel_active) {
    // For each map, threads along x-axis in a block is assigned.
    // Req.: BWD_BLK_X_SIZE < n_elems_per_channel, and
    //       n_elems_per_channel = 1 if BWD_BLK_X_SIZE < 1.
#if BWD_BLK_X_SIZE >= 2
    __shared__ float2 shm_bwd_elems_red[
        BWD_BLOCK_SIZE >= 64 ? BWD_BLOCK_SIZE + 32 : 64];
#else
    float2 *shm_bwd_elems_red;  // Dummy for compilation.
#endif

    // Loop reduction over HW.
    float val_gy_sum;
    float val_gy_x_hat_sum;

    if (is_channel_active) {
        float v1 = gy[map_cur];
        float v2 = x_hat[map_cur];
        val_gy_sum = v1;
        val_gy_x_hat_sum = v1 * v2;

        if (BWD_BLK_X_SIZE >= 1) {
            while ((map_cur += BWD_BLK_X_SIZE) < map_end) {
                float v1 = gy[map_cur];
                float v2 = x_hat[map_cur];
                val_gy_sum += v1;
                val_gy_x_hat_sum += v1 * v2;
            }
        }
    }

    // In-block reduction.
    float2 vl;

    if (is_channel_active) {
        vl = make_float2(val_gy_sum, val_gy_x_hat_sum);
        if (BWD_BLK_X_SIZE >= 2) {
            shm_bwd_elems_red[tid] = vl;
        }
    }

    if (BWD_BLK_X_SIZE >= 64) {
        __syncthreads();
    } else if (BWD_BLK_X_SIZE >= 2) {
        __syncwarp();
    }

    int is_warp_active = __any_sync(0xFFFFFFFF, is_channel_active);

#define add_to_left_bwd_elems_red(stride) add_to_left( \
    &vl, shm_bwd_elems_red, stride, \
    is_channel_active && threadIdx.x < stride || \
    is_warp_active && stride <= 32)

    if (BWD_BLK_X_SIZE >= 1024) {
        add_to_left_bwd_elems_red(512);
    }
    if (BWD_BLK_X_SIZE >= 512) {
        add_to_left_bwd_elems_red(256);
    }
    if (BWD_BLK_X_SIZE >= 256) {
        add_to_left_bwd_elems_red(128);
    }
    if (BWD_BLK_X_SIZE >= 128) {
        add_to_left_bwd_elems_red(64);
    }
    if (threadIdx.x < 32) {
        if (BWD_BLK_X_SIZE >= 64) {
            add_to_left_bwd_elems_red(32);
        }
        if (BWD_BLK_X_SIZE >= 32) {
            add_to_left_bwd_elems_red(16);
        }
        if (BWD_BLK_X_SIZE >= 16) {
            add_to_left_bwd_elems_red(8);
        }
        if (BWD_BLK_X_SIZE >= 8) {
            add_to_left_bwd_elems_red(4);
        }
        if (BWD_BLK_X_SIZE >= 4) {
            add_to_left_bwd_elems_red(2);
        }
        if (BWD_BLK_X_SIZE >= 2) {
            add_to_left_bwd_elems_red(1);
        }
    }

    // Post-map.
    if (is_channel_active && threadIdx.x == 0) {
        *gy_sum = vl.x;
        *gy_x_hat_sum = vl.y;
    }
}


extern "C"
__global__
void gn_bwd_channels_red_kernel(
        float *gy, float *x_hat, float *gamma,
        float *gy_sum_per_ch, float *gy_x_hat_sum_per_ch,
        float *gygam_avg, float *gygam_x_hat_avg,
        unsigned int n_channels_per_group, unsigned int n_elems_per_channel,
        float inv_n_elems_per_group) {
    // For each group, one block is assigned.
    __shared__ float2 shm_bwd_channels_red[
        BWD_BLK_Y_SIZE > 64 ? BWD_BLK_Y_SIZE : 64];

    // Loop reduction over channels within a group.
    // Per channel results are stored simultaneously
    // to prepare for subsequent reduction.
    unsigned int batch_offset = blockIdx.y * gridDim.x * n_channels_per_group;
    unsigned int block_id = blockIdx.y * gridDim.x + blockIdx.x;

    unsigned int ch_begin = block_id * n_channels_per_group;
    unsigned int channel_cur = ch_begin + threadIdx.y;
    unsigned int ch_end = ch_begin + n_channels_per_group;

    unsigned int map_begin = channel_cur * n_elems_per_channel;
    unsigned int map_cur = map_begin + threadIdx.x;
    unsigned int map_end = map_begin + n_elems_per_channel;

    float val_gy_sum_per_ch;
    float val_gy_x_hat_sum_per_ch;
    float val_gygam_sum = 0;
    float val_gygam_x_hat_sum = 0;

    // Settle for awkward loop structure so that all threads reaches
    // __syncthreads() while keeping inactive threads away from
    // actual computations.
    for (int is_channel_active = channel_cur < ch_end; is_channel_active; ) {
        gn_bwd_elems_red_kernel(
            gy, x_hat, &val_gy_sum_per_ch, &val_gy_x_hat_sum_per_ch,
            map_cur, map_end, is_channel_active);

        if (is_channel_active) {
            if (threadIdx.x == 0) {
                gy_sum_per_ch[channel_cur] = val_gy_sum_per_ch;
                gy_x_hat_sum_per_ch[channel_cur] = val_gy_x_hat_sum_per_ch;

                float val_gamma = gamma[channel_cur - batch_offset];
                val_gygam_sum += val_gy_sum_per_ch * val_gamma;
                val_gygam_x_hat_sum += val_gy_x_hat_sum_per_ch * val_gamma;
            }

            map_cur += BWD_BLK_Y_SIZE * n_elems_per_channel;
            map_end += BWD_BLK_Y_SIZE * n_elems_per_channel;
            channel_cur += BWD_BLK_Y_SIZE;
            is_channel_active = channel_cur < ch_end;
        }
    }

    // In-block reduction over channels within a group.
    float2 vl;
    if (threadIdx.x == 0) {
        vl = make_float2(val_gygam_sum, val_gygam_x_hat_sum);
        shm_bwd_channels_red[threadIdx.y] = vl;
    }

    if (BWD_BLOCK_SIZE >= 64) {
        __syncthreads();
    } else if (BWD_BLOCK_SIZE >= 2) {
        __syncwarp();
    }

    if (tid < BWD_BLK_Y_SIZE) {
        vl = shm_bwd_channels_red[tid];
    }

    if (BWD_BLK_Y_SIZE >= 64) {
        __syncthreads();
    } else if (BWD_BLK_Y_SIZE >= 2) {
        __syncwarp();
    }

#define add_to_left_bwd_channels_red(stride) add_to_left( \
    &vl, shm_bwd_channels_red, stride, tid < stride || stride <= 32)

    if (BWD_BLK_Y_SIZE >= 1024) {
        add_to_left_bwd_channels_red(512);
    }
    if (BWD_BLK_Y_SIZE >= 512) {
        add_to_left_bwd_channels_red(256);
    }
    if (BWD_BLK_Y_SIZE >= 256) {
        add_to_left_bwd_channels_red(128);
    }
    if (BWD_BLK_Y_SIZE >= 128) {
        add_to_left_bwd_channels_red(64);
    }
    if (tid < 32) {
        if (BWD_BLK_Y_SIZE >= 64) {
            add_to_left_bwd_channels_red(32);
        }
        if (BWD_BLK_Y_SIZE >= 32) {
            add_to_left_bwd_channels_red(16);
        }
        if (BWD_BLK_Y_SIZE >= 16) {
            add_to_left_bwd_channels_red(8);
        }
        if (BWD_BLK_Y_SIZE >= 8) {
            add_to_left_bwd_channels_red(4);
        }
        if (BWD_BLK_Y_SIZE >= 4) {
            add_to_left_bwd_channels_red(2);
        }
        if (BWD_BLK_Y_SIZE >= 2) {
            add_to_left_bwd_channels_red(1);
        }
    }

    // Post-map.
    if (tid == 0) {
        gygam_avg[block_id] = vl.x * inv_n_elems_per_group;
        gygam_x_hat_avg[block_id] = vl.y * inv_n_elems_per_group;
    }
}
'''


@cuda.memoize(for_each_device=True)
def _get_bwd_reduce_cuda_kernel(bwd_blk_x_size, bwd_blk_y_size):
    assert bwd_blk_x_size in (0,) + _REDUCE_BLOCK_SIZES
    assert bwd_blk_y_size in _REDUCE_BLOCK_SIZES
    assert bwd_blk_x_size * bwd_blk_y_size <= _REDUCE_BLOCK_SIZES[-1]
    code = string.Template(_bwd_reduce_cuda_code).substitute(
        bwd_blk_x_size=bwd_blk_x_size, bwd_blk_y_size=bwd_blk_y_size)
    prog = cuda.cupy.cuda.compile_with_cache(code)
    return prog.get_function('gn_bwd_channels_red_kernel')


def _call_bwd_reduce_cuda_kernel(gy, x_hat, gamma, shape):
    batch_size, groups, n_channels_per_group, n_elems_per_channel = shape

    gy_and_gy_x_hat_sum_per_ch = cuda.cupy.empty(
        (2, batch_size, groups, n_channels_per_group, 1), dtype=gy.dtype)
    gygam_avg = cuda.cupy.empty(batch_size * groups, dtype=gy.dtype)
    gygam_x_hat_avg = cuda.cupy.empty(batch_size * groups, dtype=gy.dtype)

    bwd_blk_x_size = 0
    for bwd_blk_x_size_candidate in _REDUCE_BLOCK_SIZES:
        if bwd_blk_x_size_candidate < n_elems_per_channel:
            bwd_blk_x_size = max(bwd_blk_x_size, bwd_blk_x_size_candidate)
    bwd_blk_y_size = _REDUCE_BLOCK_SIZES[-1] // max(bwd_blk_x_size, 1)
    for bwd_blk_y_size_candidate in _REDUCE_BLOCK_SIZES:
        if bwd_blk_y_size_candidate >= n_channels_per_group:
            bwd_blk_y_size = min(bwd_blk_y_size, bwd_blk_y_size_candidate)
    kern = _get_bwd_reduce_cuda_kernel(bwd_blk_x_size, bwd_blk_y_size)

    blocks = groups, batch_size, 1
    threads = bwd_blk_x_size, bwd_blk_y_size, 1
    kern(blocks, threads, args=(
         gy, x_hat, gamma,
         gy_and_gy_x_hat_sum_per_ch[0], gy_and_gy_x_hat_sum_per_ch[1],
         gygam_avg, gygam_x_hat_avg,
         n_channels_per_group, n_elems_per_channel,
         gy.dtype.type(1 / (n_channels_per_group * n_elems_per_channel))))

    return gy_and_gy_x_hat_sum_per_ch, gygam_avg, gygam_x_hat_avg


class GroupNormalizationGrad(function_node.FunctionNode):
    r"""The gradient of loss :math:`l` is backpropagated as follows.

    For simplicity, we assume the shape of input is :math:`(N,C,H,W)`.
    An index is denoted by :math:`\boldsymbol{i}=(i_N,i_C,i_H,i_W)`.
    We let :math:`i_G=\lfloor i_C/G \rfloor`, where :math:`G` is
    the group size.

    A set of indices :math:`S_{i_N,i_G}` is given by
    :math:`\{\boldsymbol{j}\mid j_N=i_N, j_G=i_G\}`.
    Summations are taken over such a set of indices.

    Let us see gradients on :math:`\mu` and :math:`\sigma`.

    .. math::
        \frac{\partial l}{\partial \hat{x}_\boldsymbol{i}} &=
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C}\,,
        \\
        \frac{\partial l}{\partial \mu_{i_N,i_G}} &=
        \sum_{\boldsymbol{i}\in S_{i_N,i_G}}
        \frac{\partial l}{\partial \hat{x}_\boldsymbol{i}}
        \frac{\partial \hat{x}_\boldsymbol{i}}{\partial \mu_{i_N,i_G}}
        =
        \sum_{\boldsymbol{i}\in S_{i_N,i_G}}
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C}
        (-\sigma^{-1}_{i_N,i_G})
        \\ &=
        -\sigma^{-1}_{i_N,i_G}
        \sum_{\boldsymbol{i}\in S_{i_N,i_G}}
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C}\,,
        \\
        \frac{\partial l}{\partial \sigma^2_{i_N,i_G}} &=
        \sum_{\boldsymbol{i}\in S_{i_N,i_G}}
        \frac{\partial l}{\partial \hat{x}_\boldsymbol{i}}
        \frac{\partial \hat{x}_\boldsymbol{i}}{\partial \sigma^2_{i_N,i_G}}
        =
        \sum_{\boldsymbol{i}\in S_{i_N,i_G}}
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C}
        \frac{-(x_\boldsymbol{i}-\mu_{i_N,i_G})\sigma^{-3}_{i_N,i_G}}{2}
        \\ &=
        -\frac{\sigma^{-2}_{i_N,i_G}}{2}
        \sum_{\boldsymbol{i}\in S_{i_N,i_G}}
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C}
        \hat{x}_\boldsymbol{i}\,.

    The gradient on the input :math:`x` is:

    .. math::
        \frac{\partial l}{\partial x_\boldsymbol{i}} &=
        \frac{\partial l}{\partial y_\boldsymbol{i}}
        \sigma^{-1}_{i_N,i_G}\gamma_{i_C} +
        \frac{\partial l}{\partial \mu_{i_N,i_G}}
        \frac{\partial \mu_{i_N,i_G}}{\partial x_\boldsymbol{i}} +
        \frac{\partial l}{\partial \sigma^2_{i_N,i_G}}
        \frac{\partial \sigma^2_{i_N,i_G}}{\partial x_\boldsymbol{i}}
        \\ &=
        \sigma^{-1}_{i_N,i_G}
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C} -
        \sigma^{-1}_{i_N,i_G}\Big(
        \sum_{\boldsymbol{i}\in S_{i_N,i_G}}
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C}
        \Big)
        \frac{1}{GHW}
        \\ & \phantom{{}=
        \sigma^{-1}_{i_N,i_G}
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C}}{} -
        \frac{\sigma^{-2}_{i_N,i_G}}{2}\Big(
        \sum_{\boldsymbol{i}\in S_{i_N,i_G}}
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C}
        \hat{x}_\boldsymbol{i}
        \Big)
        \Big(
        \frac{2x_\boldsymbol{i}}{GHW}-\frac{2\mu_{i_N,i_G}}{GHW}
        \Big)
        \\ &=
        \sigma^{-1}_{i_N,i_G}\Big(
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C} -
        \Big(
        \sum_{\boldsymbol{i}\in S_{i_N,i_G}}
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C}
        \Big)
        / GHW
        \\ & \phantom{{}=
        \sigma^{-1}_{i_N,i_G}\Big(
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C}}{} -
        \hat{x}_\boldsymbol{i}\Big(
        \sum_{\boldsymbol{i}\in S_{i_N,i_G}}
        \frac{\partial l}{\partial y_\boldsymbol{i}}\gamma_{i_C}
        \hat{x}_\boldsymbol{i}
        \Big)
        / GHW
        \Big)\,.

    For :math:`\gamma` and :math:`\beta`, the gradients are rather trivial.
    But efficient reduction computation becomes tricky, because summations
    here are taken over :math:`S_{i_C}=\{\boldsymbol{j} \mid j_C=i_C\}`, which
    differs to :math:`S_{i_N,i_G}`.

    .. math::
        \frac{\partial l}{\partial \gamma_{i_C}} &=
        \sum_{\boldsymbol{i}\in S_{i_C}}
        \frac{\partial l}{\partial y_\boldsymbol{i}}\hat{x}_{\boldsymbol{i}}\,,
        \\
        \frac{\partial l}{\partial \beta_{i_C}} &=
        \sum_{\boldsymbol{i}\in S_{i_C}}
        \frac{\partial l}{\partial y_\boldsymbol{i}}\,.
    """
    def __init__(self, groups, x_hat, inv_std):
        self.groups = groups
        self.x_hat = x_hat
        self.inv_std = inv_std

    def forward_cpu(self, inputs):
        gamma, gy = inputs
        x_hat, inv_std = self.x_hat, self.inv_std

        orig_shape = gy.shape
        batch_size, n_channels = orig_shape[:2]
        groups, n_channels_per_group = \
            self.groups, n_channels // self.groups
        gy = gy.reshape((batch_size, groups, n_channels_per_group, -1))
        gamma = gamma.reshape((groups, n_channels_per_group, 1))

        gy_sum_per_ch = gy.sum(axis=3, keepdims=True)
        gy_x_hat_sum_per_ch = (gy * x_hat).sum(axis=3, keepdims=True)

        inv_n_elems_per_group = 1 / numpy.prod(gy.shape[2:])
        gygam_avg = (
            (gamma * gy_sum_per_ch).sum(axis=2, keepdims=True)
            * inv_n_elems_per_group)
        gygam_x_hat_avg = (
            (gamma * gy_x_hat_sum_per_ch).sum(axis=2, keepdims=True)
            * inv_n_elems_per_group)

        gx = inv_std * (gy * gamma - gygam_avg - x_hat * gygam_x_hat_avg)
        gbeta = gy_sum_per_ch.sum(axis=0)
        ggamma = gy_x_hat_sum_per_ch.sum(axis=0)

        return (gx.reshape(orig_shape),
                ggamma.reshape((n_channels,)), gbeta.reshape((n_channels,)))

    def forward_gpu(self, inputs):
        gamma, gy = inputs
        if gy.size == 0:
            gx = cuda.cupy.empty_like(gy)
            ggamma = cuda.cupy.empty_like(gamma)
            gbeta = cuda.cupy.empty_like(gamma)
            return gx, ggamma, gbeta
        x_hat, inv_std = self.x_hat, self.inv_std

        orig_shape = gy.shape
        batch_size, n_channels = orig_shape[:2]
        n_elems_per_channel = numpy.prod(orig_shape[2:])
        groups, n_channels_per_group = \
            self.groups, n_channels // self.groups
        shape = batch_size, groups, n_channels_per_group, n_elems_per_channel

        gy = gy.ravel()
        gamma = gamma.ravel()

        gy_and_gy_x_hat_sum_per_ch, gygam_avg, gygam_x_hat_avg = \
            _call_bwd_reduce_cuda_kernel(gy, x_hat, gamma, shape)

        # Use raw indexing with specialized compilation to avoid overheads
        # caused by indexing calculation, which involves successive integer
        # divmod operations. The optimization gives massive performance gain.
        # We reuse x_hat for efficient memory access.
        gn_bwd_norm_code = string.Template('''
            unsigned int u = i;
            unsigned int tmp = u / ${n_elems_per_channel};
            unsigned int channel_idx = tmp % ${n_channels_per_group};
            tmp /= ${n_channels_per_group};
            unsigned int group_idx = tmp % ${groups};
            unsigned int batch_idx = tmp / ${groups};
            unsigned int group_norm_idx =
                batch_idx * ${groups} +
                group_idx;
            unsigned int batch_norm_idx =
                group_idx * ${n_channels_per_group} +
                channel_idx;
            x_hat[u] = inv_std[group_norm_idx] * (
                gy[u] * gamma[batch_norm_idx] -
                gygam_avg[group_norm_idx] -
                x_hat[u] * gygam_x_hat_avg[group_norm_idx]);
        ''').substitute(n_elems_per_channel=n_elems_per_channel,
                        n_channels_per_group=n_channels_per_group,
                        groups=groups)
        gn_bwd_norm_name = 'gn_bwd_norm_{}_{}_{}'.format(
                    n_elems_per_channel, n_channels_per_group, groups)
        gn_bwd_norm_kern = cuda.elementwise(
            'raw T gy, raw T gamma, raw T x_hat, raw T inv_std, '
            'raw T gygam_avg, raw T gygam_x_hat_avg', 'raw T gx',
            gn_bwd_norm_code, gn_bwd_norm_name)

        gn_bwd_norm_kern(gy, gamma, x_hat, inv_std,
                         gygam_avg, gygam_x_hat_avg,
                         size=gy.size)
        gx = x_hat
        gbeta, ggamma = gy_and_gy_x_hat_sum_per_ch.sum(axis=1)

        return (gx.reshape(orig_shape),
                ggamma.reshape((n_channels,)), gbeta.reshape((n_channels,)))

    def backward(self, indexes, grad_outputs):
        raise NotImplementedError()


def group_normalization(x, gamma, beta, groups=32, eps=1e-5):
    """Group normalization.

    This function implements a "group normalization"
    which normalizes the input units by statistics
    that are computed along the grouped channels,
    scales and shifts them.

    Args:
        x (~chainer.Variable): Batch vectors.
            Shape of this value must be `(batch_size, channels, a1, ..., ak)`,
            where `k` can be 0. Examples are an output of
            :func:`~chainer.links.Convolution2D` (`k` = 2) and that of
            :func:`~chainer.links.Linear` (`k` = 0).
        gamma (~chainer.Variable): Scaling vectors.
        beta (~chainer.Variable): Shifting vectors.

    Returns:
        ~chainer.Variable: The output variable which has the same shape as `x`.

    Note: double-backpropagation is currently unsupported.

    See: `Group Normalization <https://arxiv.org/abs/1803.08494>`_
    """
    return GroupNormalization(groups, eps).apply((x, gamma, beta))[0]
