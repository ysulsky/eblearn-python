import numpy

try:
    from numpy.lib.stride_tricks import as_strided
except ImportError:
    # this version won't work on zero-size arrays
    def as_strided(x, shape, strides):
        return numpy.ndarray.__new__(numpy.ndarray,
                                     strides=strides, shape=shape,
                                     buffer=x, dtype=x.dtype)

def unfold(x, dim, size, step):
    assert (step > 0), 'step must be positive'
    dimlen = x.shape[dim]
    n = (dimlen - size) // step
    if dimlen < size or n * step != dimlen - size:
        if dimlen > size:
            dimlen = size + n * step
        else:
            n, dimlen = -1, 0
        x = x.swapaxes(-1, dim)[...,:dimlen].swapaxes(-1, dim)
    s = x.strides[dim]
    newshape   = x.shape[:dim]   + (n+1,)    + x.shape[dim+1:]   + (size,)
    newstrides = x.strides[:dim] + (step*s,) + x.strides[dim+1:] + (s,)
    return as_strided(x, shape=newshape, strides=newstrides)

def select(x, dim, idx):
    if dim == 0: return x[idx]
    ret = x.swapaxes(0,dim)[idx]
    transpose=[dim-1] + range(dim-1) + range(dim,ret.ndim)
    return ret.transpose(transpose)

def narrow(x, dim, size, offset = 0):
    return x.swapaxes(0,dim)[offset:offset+size].swapaxes(0,dim)

def reverse(x):
    strides, shape = x.strides, x.shape
    for i in xrange(x.ndim):
        if x.shape[i] > 0:
            x = narrow(x,i,1,shape[i]-1)
    return as_strided(x, shape, [-s for s in strides])

def reverse_along(x, dim):
    shape, strides = x.shape, x.strides
    d = shape[dim]
    if d < 1: return x
    x = narrow(x, dim, 1, d-1)
    strides = list(strides)
    strides[dim] = -strides[dim]
    return as_strided(x, shape, strides)

try:
    from gofast.idx import *
except ImportError:
    pass
