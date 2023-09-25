# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from random import random

r"""
    https://blog.bruce-hill.com/a-faster-weighted-random-choice
"""
def prepare_aliased_randomizer(weights):
    N = len(weights)
    avg = sum(weights)/N
    aliases = [(1, None)]*N
    smalls = ((i, w/avg) for i,w in enumerate(weights) if w < avg)
    bigs = ((i, w/avg) for i,w in enumerate(weights) if w >= avg)
    small, big = next(smalls, None), next(bigs, None)
    while big and small:
        aliases[small[0]] = (small[1], big[0])
        big = (big[0], big[1] - (1-small[1]))
        if big[1] < 1:
            small = big
            big = next(bigs, None)
        else:
            small = next(smalls, None)

    def weighted_random(): 
        r = random()*N
        i = int(r)
        odds, alias = aliases[i]
        return alias if (r-i) > odds else i

    return weighted_random

if __name__ == '__main__':
    import numpy as np
    from collections import defaultdict
    weights = np.arange(10, dtype=np.float32)
    weights /= np.sum(weights)

    sampler = prepare_aliased_randomizer(weights)

    n = 10000 
    res01, res02, res03 = defaultdict(int), defaultdict(int), defaultdict(int)
    for _ in range(n):
        idx01 = sampler()
        idx02 = np.random.choice(10, size=1, replace=True, p=weights)[0]
        idx03 = np.random.choice(10, size=1, replace=True)[0]
        res01[idx01] += 1
        res02[idx02] += 1
        res03[idx03] += 1
    print(sorted(res01.items(), key=lambda t:t[1]))
    print(sorted(res02.items(), key=lambda t:t[1]))
    print(sorted(res03.items(), key=lambda t:t[1]))