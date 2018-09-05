# To the extent possible under law, ALBERT Inc. has waived all copyright and
# related or neighboring rights to this work.
# For legal code, see ``CC0.txt`` distributed with this file or
# <https://creativecommons.org/publicdomain/zero/1.0/>_.
#
# We thank to the original author Sebastiano Vigna.
from collections import namedtuple


__all__ = ['Random']


class Random(namedtuple('Random', ('s0', 's1', 's2', 's3'))):
    """Xoshiro128+ PRNG implementation with minimal and immutable API.

    This implementation is a port from
    <http://vigna.di.unimi.it/xorshift/xoshiro128plus.c>_.
    """
    @classmethod
    def from_python_random(cls):
        import random
        return cls(*(random.randrange(0, 2 ** 32) for _ in range(4)))

    @classmethod
    def from_numpy_random(cls, random_state=None):
        if random_state is None:
            import numpy.random
            random_state = numpy.random
        seeds = random_state.randint(-2 ** 31, 2 ** 31, 4).astype('uint32')
        return cls(*map(int, seeds))

    def gen(self):
        s0, s1, s2, s3 = self
        result_plus = (s0 + s3) & 0xFFFFFFFF
        t = (s1 << 9) & 0xFFFFFFFF
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = ((s3 << 11) | (s3 >> 21)) & 0xFFFFFFFF
        return result_plus, self.__class__(s0, s1, s2, s3)

    def jump(self):
        jump = 0x8764000B, 0xF542D2D3, 0x6FA035C3, 0x77F2DB5B
        s0, s1, s2, s3 = 0, 0, 0, 0
        for j in jump:
            for b in range(32):
                if j & (1 << b):
                    s0 ^= self.s0
                    s1 ^= self.s1
                    s2 ^= self.s2
                    s3 ^= self.s3
                _, self = self.gen()
        return self.__class__(s0, s1, s2, s3)
