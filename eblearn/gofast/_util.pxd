cimport cython
cimport numpy as np

cdef extern from "numpy/arrayobject.h":
    cdef void import_array()
    cdef bint     PyArray_ISCONTIGUOUS(np.ndarray)
    cdef bint     PyArray_ISBEHAVED(np.ndarray)
    cdef bint     PyArray_ISCARRAY(np.ndarray)
    cdef bint     PyArray_ISCARRAY_RO(np.ndarray)
    cdef bint     PyArray_ISWRITEABLE(np.ndarray)
    cdef bint     PyArray_SAMESHAPE(np.ndarray, np.ndarray)
    cdef long     PyArray_SIZE(np.ndarray)
    cdef long     PyArray_NBYTES(np.ndarray)
    cdef int      PyArray_ITEMSIZE(np.ndarray)
    cdef int      PyArray_TYPE(np.ndarray)
    cdef np.ndarray PyArray_ZEROS(int, np.npy_intp *dims, int, bint)
    cdef np.ndarray PyArray_EMPTY(int, np.npy_intp *dims, int, bint)
    cdef void     PyArray_UpdateFlags(np.ndarray, int)
    

cdef extern from "math.h":
    cdef double  exp(double)
    cdef double  sqrt(double)
    cdef double  log(double)
    cdef double  tanh(double)

cdef extern from "string.h":
    cdef void*   memset(void*, int, size_t)


IF 1:
    ctypedef np.float64_t rtype_t
    cdef enum:
        NPY_RTYPE = np.NPY_DOUBLE
ELSE:
    ctypedef np.float32_t rtype_t
    cdef enum:
        NPY_RTYPE = np.NPY_FLOAT

