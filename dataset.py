import bisect
import ctypes
import os
import os.path
from multiprocessing import sharedctypes

import numpy as np
from chainer.dataset import DatasetMixin
from PIL import Image

from xoshiro import Random


__all__ = ['short_edge_resize', 'randnat_below', 'random_crop',
           'random_horizontal_flip', 'center_crop', 'CatsDataset']


def short_edge_resize(image, short_edge):
    old_w, old_h = image.size
    if old_w > old_h:
        new_w = old_w * short_edge // old_h
        new_h = short_edge
    else:
        new_w = short_edge
        new_h = old_h * short_edge // old_w
    return image.resize((new_w, new_h), resample=Image.BILINEAR)


def randnat_below(sup, random):
    if not 0 < sup <= 2 ** 32:
        raise ValueError
    scale = (2 ** 32) // sup
    while True:
        nat, random = random.gen()
        nat //= scale
        if nat < sup:
            break
    return nat, random


def random_crop(image, random, crop_edge):
    w, h = image.size
    shift_x_sup = w - crop_edge + 1
    shift_y_sup = h - crop_edge + 1
    shift_x, random = randnat_below(shift_x_sup, random)
    shift_y, random = randnat_below(shift_y_sup, random)
    image = image.crop((shift_x, shift_y,
                        shift_x + crop_edge, shift_y + crop_edge))
    return image, random


def random_horizontal_flip(image, random):
    r, random = random.gen()
    if r >> 31:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image, random


def center_crop(image, crop_edge):
    w, h = image.size
    shift_x = (w - crop_edge) // 2
    shift_y = (h - crop_edge) // 2
    image = image.crop((shift_x, shift_y,
                        shift_x + crop_edge, shift_y + crop_edge))
    return image


class CatsDataset(DatasetMixin):
    """Cats classification dataset implementation for VGG pets dataset.

    .. warning::
        Overlapping batch elements will cause undesirable consequence.
        This is because a dataset instance stores xoshiro128+ PRNG states
        on shared memory in order to achieve reproducible image-augmentation
        with :class:`MultiprocessIterator` support.
    """
    classes = (
        'Abyssinian',
        'Bengal',
        'Birman',
        'Bombay',
        'British_Shorthair',
        'Egyptian_Mau',
        'Maine_Coon',
        'Persian',
        'Ragdoll',
        'Russian_Blue',
        'Siamese',
        'Sphynx',
    )

    train_ratio = 0.8

    def __init__(self, image_dir, short_edge, crop_edge,
                 filenames, randoms, train):
        if short_edge < crop_edge:
            raise ValueError
        self._image_dir = image_dir
        self._short_edge = short_edge
        self._crop_edge = crop_edge
        self._filenames = filenames
        self._train = train
        if train:
            randoms = np.asarray(randoms, dtype=np.uint32).ravel()
            self._randoms = sharedctypes.Array(ctypes.c_uint32, randoms)

        self._class_seps = []
        t = 0
        for i, filename in enumerate(filenames):
            if not filename.startswith(self.classes[t]):
                t += 1
                if i == 0 or not filename.startswith(self.classes[t]):
                    raise ValueError
                self._class_seps.append(i)
        if t != len(self.classes) - 1:
            raise ValueError

    @classmethod
    def train_valid(cls, image_dir, short_edge, crop_edge, random):
        filenames = [filename for filename in os.listdir(image_dir)
                     if (filename.startswith(cls.classes) and
                         filename.endswith('.jpg'))]
        filenames.sort()

        train_thresh = int((2 ** 32) * cls.train_ratio)
        train_filenames = []
        valid_filenames = []
        for filename in filenames:
            r, random = random.gen()
            if r <= train_thresh:
                train_filenames.append(filename)
            else:
                valid_filenames.append(filename)

        randoms = []
        for _ in train_filenames:
            randoms.append(random)
            random = random.jump()

        train_dataset = cls(image_dir, short_edge, crop_edge, train_filenames,
                            randoms, True)
        valid_dataset = cls(image_dir, short_edge, crop_edge, valid_filenames,
                            None, False)
        return train_dataset, valid_dataset, random

    def get_example(self, i):
        image = Image.open(os.path.join(self._image_dir, self._filenames[i]))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = short_edge_resize(image, self._short_edge)
        if self._train:
            random = Random(*self._randoms[4*i:4*(i+1)])
            image, random = random_crop(image, random, self._crop_edge)
            image, random = random_horizontal_flip(image, random)
            self._randoms[4*i:4*(i+1)] = random
        else:
            image = center_crop(image, self._crop_edge)

        image = np.rollaxis(np.asarray(image), 2).astype(np.float32) - 127.5
        t = bisect.bisect(self._class_seps, i)
        return image, t

    def __len__(self):
        return len(self._filenames)
