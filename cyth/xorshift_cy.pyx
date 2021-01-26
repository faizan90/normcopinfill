# cython: nonecheck=False, boundscheck=False, wraparound=False, cdivision=True
import sys
import time

import numpy as np
cimport numpy as np

ctypedef double DT_D
ctypedef unsigned long DT_UL

# start a random number sequence
# xorshift prng
cdef:
    DT_D MOD_long_long = sys.maxsize * 2.
    unsigned long long SEED = time.time() * 1000
    unsigned long long rnd_i, A, B, C

rnd_i = SEED
A = 21
B = 35
C = 4


cdef DT_D rand_c() nogil:
    global rnd_i
    rnd_i ^= rnd_i << A
    rnd_i ^= rnd_i >> B
    rnd_i ^= rnd_i << C
    return rnd_i / MOD_long_long


# warmup
cdef DT_UL _
for _ in xrange(1000):
    rand_c()


cpdef gen_n_rns_arr(DT_UL n_rns):
    cdef:
        DT_UL i
        np.ndarray[DT_D, ndim=1] rns_arr = np.zeros(n_rns, dtype=np.float64)

    for i in xrange(n_rns):
        rns_arr[i] = rand_c()

    return rns_arr

