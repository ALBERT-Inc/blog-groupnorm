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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)


if __name__ == '__main__':
    main()
