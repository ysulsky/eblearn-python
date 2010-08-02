cimport numpy as np
from util cimport rtype_t

cdef c_dtanh (rtype_t *x, long n, rtype_t *r)
cdef c_ddtanh(rtype_t *x, long n, rtype_t *r)

cdef rtype_t c_m1ldot(char *a, char *b,
                      int n,
                      int a_s0, int b_s0)
cdef rtype_t c_m2ldot(char *a,  char *b,
                      int n0,   int n1,
                      int a_s0, int a_s1,
                      int b_s0, int b_s1)
cdef rtype_t c_m3ldot(char *a,  char *b,
                      int n0,   int n1,   int n2,
                      int a_s0, int a_s1, int a_s2,
                      int b_s0, int b_s1, int b_s2)

cdef np.ndarray c_m1extm1(np.ndarray m1, np.ndarray m2,
                          np.ndarray res, bint accumulate)
cdef np.ndarray c_m2extm2(np.ndarray m1, np.ndarray m2,
                          np.ndarray res, bint accumulate)
cdef np.ndarray c_m3extm3(np.ndarray m1, np.ndarray m2,
                          np.ndarray res, bint accumulate)

cdef np.ndarray c_m2dotm1(np.ndarray m1, np.ndarray m2,
                          np.ndarray res, bint accumulate)
cdef np.ndarray c_m4dotm2(np.ndarray m1, np.ndarray m2,
                          np.ndarray res, bint accumulate)
cdef np.ndarray c_m6dotm3(np.ndarray m1, np.ndarray m2,
                          np.ndarray res, bint accumulate)

