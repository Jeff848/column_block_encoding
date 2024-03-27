import numpy as np

def get_padded_matrix(a):
    n, m = a.shape
    if n != m:
        k = max(n, m)
        a = np.pad(a, ((0, k - n), (0, k - m)))
        n = k
    logn = int(np.ceil(np.log2(n)))
    if n < 2**logn:
        a = np.pad(a, ((0, 2**logn - n), (0, 2**logn - n)))
        n = 2**logn
    return a, n, logn

def gen_random_positive_int_matrix():
    return 1

def gen_random_snp_matrix():
    return 1