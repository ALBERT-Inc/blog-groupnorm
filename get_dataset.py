#!/usr/bin/env python
import hashlib
import os.path
import shutil
import tarfile
from urllib.request import urlopen


def download(url, md5_hexdigest):
    filename = url.split('/')[-1]

    if not os.path.exists(filename):
        with urlopen(url) as resp, open(filename, 'wb') as f:
            shutil.copyfileobj(resp, f)

    with open(filename, 'rb') as f:
        m = hashlib.md5()
        bufsize = 1024 * 1024
        while True:
            buf = f.read(bufsize)
            if not buf:
                break
            m.update(buf)
        if m.hexdigest() != md5_hexdigest:
            raise ValueError('corrupted file')

    return filename


IMAGES_URL = 'http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
IMAGES_MD5 = '5c4f3ee8e5d25df40f4fd59a7f44e54c'


def main():
    filename = download(IMAGES_URL, IMAGES_MD5)
    with tarfile.open(filename) as tar:
        tar.extractall()


if __name__ == '__main__':
    main()
