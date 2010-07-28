cimport cython
cimport numpy as np

cdef extern from "numpy/arrayobject.h":
    cdef void import_array()
    cdef bint     PyArray_ISCONTIGUOUS(np.ndarray)
    cdef bint     PyArray_SAMESHAPE(np.ndarray, np.ndarray)
    cdef long     PyArray_SIZE(np.ndarray)
    cdef int      PyArray_ITEMSIZE(np.ndarray)

cdef extern from "math.h":
    cdef double  exp(double)
    cdef double  sqrt(double)
    cdef double  log(double)
    cdef double  tanh(double)

cdef extern from "string.h":
    cdef void*   memset(void*, int, size_t)


# also change me in _util.pyx
ctypedef np.float64_t rtype_t

