#!/usr/bin/env python
import argparse
import os
import os.path

import chainer
import numpy.random
from chainer import iterators, optimizer_hooks, optimizers, training
from chainer.training import extensions

from dataset import CatsDataset
from resnet import ResNet50
from xoshiro import Random


NORMALIZATIONS = (
    'bn',
    'gnchainer',
    'gnalb1',
    'gnalb2',
)


def get_normalization(name):
    if name not in NORMALIZATIONS:
        raise ValueError

    if name == 'bn':
        from chainer.links import BatchNormalization
        return BatchNormalization

    if name == 'gnchainer':
        from chainer.links import GroupNormalization as GNChainer

        def gn_chainer(size, eps):
            return GNChainer(groups=32, size=size, eps=eps)

        return gn_chainer

    if name == 'gnalb1':
        from group_normalization_alb_link import GroupNormalizationAlb1
        return GroupNormalizationAlb1

    if name == 'gnalb2':
        from group_normalization_alb_link import GroupNormalizationAlb2
        return GroupNormalizationAlb2

    assert False


def main():
    import multiprocessing
    multiprocessing.set_start_method('forkserver')

    parser = argparse.ArgumentParser(description='Cats training.')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--normalization', type=str, choices=NORMALIZATIONS,
                        required=True,
                        help='Normalization method')
    args = parser.parse_args()

    gpu = args.gpu
    out_dir = args.out
    image_dir = 'images'

    batch_size = 32
    short_edge = 256
    crop_edge = 224

    seed = 3141592653
    n_processes = len(os.sched_getaffinity(0))

    normalization = get_normalization(args.normalization)

    initial_lr = 0.1
    epochs = 300
    lr_reduce_interval = (100, 'epoch')
    lr_reduce_rate = 0.1
    weight_decay = 5e-4

    numpy_random = numpy.random.RandomState(seed)
    random = Random.from_numpy_random(numpy_random)
    train_dataset, valid_dataset, _ = CatsDataset.train_valid(
        image_dir, short_edge, crop_edge, random)
    order_sampler = iterators.ShuffleOrderSampler(numpy_random)
    train_iter = iterators.MultiprocessIterator(train_dataset, batch_size,
                                                repeat=True, shuffle=None,
                                                n_processes=n_processes,
                                                n_prefetch=4,
                                                order_sampler=order_sampler)
    valid_iter = iterators.MultiprocessIterator(valid_dataset, batch_size,
                                                repeat=False, shuffle=False,
                                                n_processes=n_processes,
                                                n_prefetch=4)

    numpy.random.seed(seed)
    model = ResNet50(len(CatsDataset.classes), normalization)
    model = chainer.links.Classifier(model)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    optimizer = optimizers.MomentumSGD(lr=initial_lr)
    optimizer.setup(model)
    optimizer.add_hook(optimizer_hooks.WeightDecay(weight_decay))

    updater = training.updaters.StandardUpdater(train_iter, optimizer,
                                                device=gpu)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out=out_dir)

    trainer.extend(extensions.ExponentialShift('lr', lr_reduce_rate),
                   trigger=lr_reduce_interval)
    trainer.extend(extensions.Evaluator(valid_iter, model, device=gpu),
                   trigger=(1, 'epoch'))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.run()

    chainer.serializers.save_npz(os.path.join(out_dir, 'model.npz'), model)


if __name__ == '__main__':
    main()
