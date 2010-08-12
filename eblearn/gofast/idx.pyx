# mode -*-python-*-

cimport cython

from eblearn.gofast.util cimport *

import_array()

@cython.boundscheck(False)
def unfold(np.ndarray x, int dim, int size, int step):
    cdef int dimlen, n, s, i
    assert (size > 0 and step > 0), 'size and step must be positive'
    assert (x.ndim > dim >= 0), 'dim is out of bounds'
    x = x[None] # add another dimension, don't modify x (sets non-contig flag)
    dim   += 1
    dimlen = x.shape[dim]
    n = (dimlen - size) / step
    if dimlen < size or n*step != dimlen - size:
        if dimlen > size:
            dimlen = size + n*step
        else:
            n, dimlen = -1, 0
        x.shape[dim] = dimlen
    s = x.strides[dim]
    dim -= 1
    for i in range(dim):
        x.shape[i]   = x.shape[i+1]
        x.strides[i] = x.strides[i+1]
    x.shape[dim]   = n+1
    x.strides[dim] = step*s
    for i in range(dim+1,x.ndim-1):
        x.shape[i]   = x.shape[i+1]
        x.strides[i] = x.strides[i+1]
    x.shape[x.ndim-1]   = size
    x.strides[x.ndim-1] = s
    return x

@cython.boundscheck(False)
def select(np.ndarray x, int dim, int idx):
    cdef int i, stride, size
    cdef np.ndarray ret
    assert (x.ndim > dim >= 0 and x.shape[dim] > idx >= 0),'index out of bounds'
    if dim == 0:        return x[idx]
    stride = x.strides[0]
    size   = x.shape[0]
    x.strides[0]   = x.strides[dim]
    x.shape[0]     = x.shape[dim]
    x.strides[dim] = stride
    x.shape[dim]   = size
    ret = x[dim]
    x.strides[dim] = x.strides[0]
    x.shape[dim]   = x.shape[0]
    x.strides[0]   = stride
    x.shape[0]     = size
    dim -= 1
    if dim > 0:
        for i in range(dim, 0, -1):
            ret.strides[i] = ret.strides[i-1]
            ret.shape[i]   = ret.shape[i-1]
        ret.strides[0] = stride
        ret.shape[0] = size
    PyArray_UpdateFlags(ret, NPY_C_CONTIGUOUS)
    return ret


@cython.boundscheck(False)
def narrow(np.ndarray x, int dim, int size, int offset = 0):
    assert (x.ndim > dim >= 0 and offset >= 0 and size >= 0
            and offset+size <= x.shape[dim]), "indices out of bounds"
    x = x[:]
    x.shape[dim] = size
    if offset:
        x.data += offset * x.strides[dim]
    PyArray_UpdateFlags(x, NPY_C_CONTIGUOUS)
    return x

@cython.boundscheck(False)
def reverse(np.ndarray x):
    cdef int i
    cdef long n = 0
    for i in range(x.ndim):
        if x.shape[i] > 0:
            n += (x.shape[i] - 1) * x.strides[i]
    x = x[:]
    x.data += n
    for i in range(x.ndim):
        x.strides[i] = -x.strides[i]
    PyArray_UpdateFlags(x, NPY_C_CONTIGUOUS)
    return x

@cython.boundscheck(False)
def reverse_along(np.ndarray x, int dim):
    assert (0 <= dim < x.ndim), "dimension out of bounds"
    if x.shape[dim] < 1: return x
    x = x[:]
    x.data += (x.shape[dim] - 1) * x.strides[dim]
    x.strides[dim] = -x.strides[dim]
    PyArray_UpdateFlags(x, NPY_C_CONTIGUOUS)
    return x

