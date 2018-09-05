import functools

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import HeNormal


__all__ = ['ResNet50']


class Res1(chainer.Chain):
    """Res1 link.

    Args:
        output_channels (int): Number of channels of output arrays.
        normalization (callable): Constructor of normalization layers.
    """
    def __init__(self, output_channels, normalization):
        super().__init__()
        with self.init_scope():
            # Original ResNet-50 has a bias term in conv1, but ResNet-101 and
            # ResNet-152 don't. The original bias term is likely a mistake, so
            # this implementation omits it.
            # https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt
            self.conv = L.Convolution2D(None, output_channels,
                                        ksize=7, stride=2, pad=3,
                                        initialW=HeNormal(), nobias=True)
            self.norm = normalization(output_channels, eps=1e-5)

    def __call__(self, x):
        h = F.relu(self.norm(self.conv(x)))
        return F.max_pooling_2d(h, 3, stride=2)


class BottleneckBlockA(chainer.Chain):
    """Bottleneck block that reduces the resolution of the feature map.

    Args:
        input_channels (int): Number of channels of input arrays.
        inner_channels (int): Number of channels of intermediate arrays.
        output_channels (int): Number of channels of output arrays.
        normalization (callable): Constructor of normalization layers.
        stride (int or tuple of ints): Stride of filter application.
    """
    def __init__(self, input_channels, inner_channels, output_channels,
                 normalization, stride=2):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(input_channels, inner_channels,
                                         ksize=1, stride=stride, pad=0,
                                         initialW=HeNormal(), nobias=True)
            self.norm1 = normalization(inner_channels, eps=1e-5)
            self.conv2 = L.Convolution2D(inner_channels, inner_channels,
                                         ksize=3, stride=1, pad=1,
                                         initialW=HeNormal(), nobias=True)
            self.norm2 = normalization(inner_channels, eps=1e-5)
            self.conv3 = L.Convolution2D(inner_channels, output_channels,
                                         ksize=1, stride=1, pad=0,
                                         initialW=HeNormal(), nobias=True)
            self.norm3 = normalization(output_channels, eps=1e-5)
            self.conv4 = L.Convolution2D(input_channels, output_channels,
                                         ksize=1, stride=stride, pad=0,
                                         initialW=HeNormal(), nobias=True)
            self.norm4 = normalization(output_channels, eps=1e-5)

    def __call__(self, x):
        h1 = F.relu(self.norm1(self.conv1(x)))
        h1 = F.relu(self.norm2(self.conv2(h1)))
        h1 = self.norm3(self.conv3(h1))
        h2 = self.norm4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleneckBlockB(chainer.Chain):
    """Bottleneck block that maintains the resolution of the feature map.

    Args:
        input_channels (int): Number of channels of input arrays.
        inner_channels (int): Number of channels of intermediate arrays.
        normalization (callable): Constructor of normalization layers.
    """
    def __init__(self, input_channels, inner_channels, normalization):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(input_channels, inner_channels,
                                         ksize=1, stride=1, pad=0,
                                         initialW=HeNormal(), nobias=True)
            self.norm1 = normalization(inner_channels, eps=1e-5)
            self.conv2 = L.Convolution2D(inner_channels, inner_channels,
                                         ksize=3, stride=1, pad=1,
                                         initialW=HeNormal(), nobias=True)
            self.norm2 = normalization(inner_channels, eps=1e-5)
            self.conv3 = L.Convolution2D(inner_channels, input_channels,
                                         ksize=1, stride=1, pad=0,
                                         initialW=HeNormal(), nobias=True)
            self.norm3 = normalization(input_channels, eps=1e-5)

    def __call__(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = F.relu(self.norm2(self.conv2(h)))
        h = self.norm3(self.conv3(h))
        return F.relu(h + x)


class ResidualLayer(chainer.Sequential):
    """Residual layer consisted of several Bottleneck blocks.

    Args:
        n_blocks (int): Number of blocks used in the layer.
        input_channels (int): Number of channels of input arrays.
        inner_channels (int): Number of channels of intermediate arrays.
        output_channels (int): Number of channels of output arrays.
        normalization (callable): Constructor of normalization layers.
        stride (int or tuple of ints): Stride of filter application.
    """
    def __init__(self, n_blocks, input_channels, inner_channels,
                 output_channels, normalization, stride=2):
        layers = []
        for i in range(n_blocks):
            if i == 0:
                b = BottleneckBlockA(input_channels, inner_channels,
                                     output_channels, normalization,
                                     stride=stride)
            else:
                b = BottleneckBlockB(output_channels, inner_channels,
                                     normalization)
            layers.append(b)
        super().__init__(*layers)


class ResNet50(chainer.Sequential):
    """ResNet-50 with pluggable normalization routine.

    Args:
        n_classes (int): Number of classes.
        normalization (callable): Constructor of normalization layers.
    """
    def __init__(self, n_classes, normalization):
        if n_classes <= 1:
            raise ValueError('n_classes must be larger than 1')

        super().__init__(
            Res1(64, normalization),
            ResidualLayer(3, 64, 64, 256, normalization, stride=1),
            ResidualLayer(4, 256, 128, 512, normalization),
            ResidualLayer(6, 512, 256, 1024, normalization),
            ResidualLayer(3, 1024, 512, 2048, normalization),
            functools.partial(F.average, axis=(2, 3)),
            L.Linear(2048, n_classes))
