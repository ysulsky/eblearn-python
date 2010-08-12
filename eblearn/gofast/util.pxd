cimport numpy as np

cdef extern from "Python.h":
    void Py_INCREF(object)
    void Py_XINCREF(object)
    void Py_DECREF(object)
    void Py_XDECREF(object)
    str PyString_FromString(char*)

cdef extern from "numpy/arrayobject.h":
    void import_array()

    cdef enum:
         # missing in Cython < 0.12
         NPY_CONTIGUOUS, NPY_C_CONTIGUOUS,
         NPY_UPDATEIFCOPY, NPY_WRITEABLE,
         NPY_FORCECAST

    np.dtype PyArray_DescrFromType(int) # also missing from Cython < 0.12
    
    bint     PyArray_ISCONTIGUOUS(np.ndarray)
    bint     PyArray_ISBEHAVED(np.ndarray)
    bint     PyArray_ISCARRAY(np.ndarray)
    bint     PyArray_ISCARRAY_RO(np.ndarray)
    bint     PyArray_ISWRITEABLE(np.ndarray)
    bint     PyArray_ISONESEGMENT(np.ndarray)
    bint     PyArray_FILLWBYTE(np.ndarray, int)
    bint     PyArray_SAMESHAPE(np.ndarray, np.ndarray)
    long     PyArray_SIZE(np.ndarray)
    long     PyArray_NBYTES(np.ndarray)
    int      PyArray_ITEMSIZE(np.ndarray)
    int      PyArray_TYPE(np.ndarray)
    np.ndarray PyArray_ZEROS(int, np.npy_intp *dims, int, bint)
    np.ndarray PyArray_EMPTY(int, np.npy_intp *dims, int, bint)
    void     PyArray_UpdateFlags(np.ndarray, int)
    
    int      PyArray_CopyInto(np.ndarray dest, np.ndarray src)
    int      PyArray_MoveInto(np.ndarray dest, np.ndarray src)
    
    # this is slower than going through python 
    # (NumPy replaces np.dot but not this)
    np.ndarray PyArray_InnerProduct(np.ndarray, np.ndarray)
    
    # steals a reference from dtype
    np.ndarray PyArray_FromArray(np.ndarray, np.dtype, int) 

ctypedef struct npy_intp2:
     np.npy_intp d0, d1

ctypedef struct npy_intp3:
     np.npy_intp d0, d1, d2

ctypedef struct npy_intp4:
     np.npy_intp d0, d1, d2, d3

ctypedef struct npy_intp5:
     np.npy_intp d0, d1, d2, d3, d4

ctypedef struct npy_intp6:
     np.npy_intp d0, d1, d2, d3, d4, d5

cdef inline PyArray_EMPTY1(np.npy_intp d0, int t):
     return PyArray_EMPTY(1, &d0, t, 0)

cdef inline PyArray_ZEROS1(np.npy_intp d0, int t):
     return PyArray_ZEROS(1, &d0, t, 0)

cdef inline PyArray_EMPTY2(np.npy_intp d0, np.npy_intp d1, int t):
     cdef npy_intp2 d = {'d0':d0, 'd1':    d1}
     return PyArray_EMPTY(2, <np.npy_intp*>&d, t, 0)
     
cdef inline PyArray_ZEROS2(np.npy_intp d0, np.npy_intp d1, int t):
     cdef npy_intp2 d = {'d0':d0, 'd1':d1}
     return PyArray_ZEROS(2, <np.npy_intp*>&d, t, 0)

cdef inline PyArray_EMPTY3(np.npy_intp d0, np.npy_intp d1, np.npy_intp d2, 
                           int t):
     cdef npy_intp3 d = {'d0':d0, 'd1':d1, 'd2':d2}
     return PyArray_EMPTY(3, <np.npy_intp*>&d, t, 0)

cdef inline PyArray_ZEROS3(np.npy_intp d0, np.npy_intp d1, np.npy_intp d2, 
                           int t):
     cdef npy_intp3 d = {'d0':d0, 'd1':d1, 'd2':d2}
     return PyArray_ZEROS(3, <np.npy_intp*>&d, t, 0)

cdef inline PyArray_EMPTY4(np.npy_intp d0, np.npy_intp d1, np.npy_intp d2,
                           np.npy_intp d3, int t):
     cdef npy_intp4 d = {'d0':d0, 'd1':d1, 'd2':d2, 'd3':d3}
     return PyArray_EMPTY(4, <np.npy_intp*>&d, t, 0)

cdef inline PyArray_ZEROS4(np.npy_intp d0, np.npy_intp d1, np.npy_intp d2,
                           np.npy_intp d3, int t):
     cdef npy_intp4 d = {'d0':d0, 'd1':d1, 'd2':d2, 'd3':d3}
     return PyArray_ZEROS(4, <np.npy_intp*>&d, t, 0)

cdef inline PyArray_EMPTY5(np.npy_intp d0, np.npy_intp d1, np.npy_intp d2,
                           np.npy_intp d3, np.npy_intp d4, int t):
     cdef npy_intp5 d = {'d0':d0, 'd1':d1, 'd2':d2, 'd3':d3, 'd4':d4}
     return PyArray_EMPTY(5, <np.npy_intp*>&d, t, 0)

cdef inline PyArray_ZEROS5(np.npy_intp d0, np.npy_intp d1, np.npy_intp d2,
                           np.npy_intp d3, np.npy_intp d4, int t):
     cdef npy_intp5 d = {'d0':d0, 'd1':d1, 'd2':d2, 'd3':d3, 'd4':d4}
     return PyArray_ZEROS(5, <np.npy_intp*>&d, t, 0)

cdef inline PyArray_EMPTY6(np.npy_intp d0, np.npy_intp d1, np.npy_intp d2,
                           np.npy_intp d3, np.npy_intp d4, np.npy_intp d5,
                           int t):
     cdef npy_intp6 d = {'d0':d0, 'd1':d1, 'd2':d2, 'd3':d3, 'd4':d4, 'd5':d5}
     return PyArray_EMPTY(6, <np.npy_intp*>&d, t, 0)

cdef inline PyArray_ZEROS6(np.npy_intp d0, np.npy_intp d1, np.npy_intp d2,
                           np.npy_intp d3, np.npy_intp d4, np.npy_intp d5,
                           int t):
     cdef npy_intp6 d = {'d0':d0, 'd1':d1, 'd2':d2, 'd3':d3, 'd4':d4, 'd5':d5}
     return PyArray_ZEROS(6, <np.npy_intp*>&d, t, 0)

cdef enum:
    RESULTFLAGS = NPY_UPDATEIFCOPY | NPY_WRITEABLE

cdef inline np.ndarray cvt(np.ndarray m, int npy_type, int flags):
    cdef np.dtype t
    if m is None: return None
    t = PyArray_DescrFromType(npy_type)
    flags |= NPY_FORCECAST
    Py_INCREF(t) # the following steals a reference
    IF 1:
        return PyArray_FromArray(m, t, flags)
    ELSE:
        m2 = PyArray_FromArray(m, t, flags)
        if m2 is not m: print '+++ Warning: conversion happening'
        return m2


cdef extern from "math.h":
    double  exp(double)
    double  sqrt(double)
    double  log(double)
    double  tanh(double)

cdef extern from "string.h":
    void*   memset(void*, int, size_t)


IF 1:
    ctypedef np.float64_t rtype_t
    cdef enum:
        NPY_RTYPE = np.NPY_DOUBLE
ELSE:
    ctypedef np.float32_t rtype_t
    cdef enum:
        NPY_RTYPE = np.NPY_FLOAT

