# mode -*-python-*-

cimport cython

import  numpy as np
cimport numpy as np

cdef extern from "numpy/arrayobject.h":
    cdef void import_array()

import_array()

rtype = np.float64
ctypedef np.float64_t rtype_t


@cython.boundscheck(False)
def correlate_2d_valid_simple(np.ndarray[rtype_t, ndim=2] input  not None,
                       np.ndarray[rtype_t, ndim=2] kernel not None,
                       np.ndarray[rtype_t, ndim=2] output = None):

    cdef int kymax = kernel.shape[0]
    cdef int kxmax = kernel.shape[1]
    cdef int ymax = input.shape[0] + 1 - kymax
    cdef int xmax = input.shape[1] + 1 - kxmax
    
    cdef int y, x, ky, kx
    cdef rtype_t val
    
    if output is None:
        output = np.empty((ymax, xmax), dtype=rtype)
    
    assert (output.shape[0] == ymax and output.shape[1] == xmax)
    
    for y in range(ymax):
        for x in range(xmax):
            val = 0.
            
            for ky in range(kymax):
                for kx in range(kxmax):
                    val += input[y + ky, x + kx] * kernel[ky, kx]

            output[y, x] = val

    return output

@cython.boundscheck(False)
def correlate_2d_valid(np.ndarray[rtype_t, ndim=2] input  not None,
                       np.ndarray[rtype_t, ndim=2] kernel not None,
                       np.ndarray[rtype_t, ndim=2] output = None):

    cdef int kymax = kernel.shape[0]
    cdef int kxmax = kernel.shape[1]
    cdef int ymax = input.shape[0] + 1 - kymax
    cdef int xmax = input.shape[1] + 1 - kxmax
    
    cdef int y, x, ky, kx
    cdef int sy, sx, soy, sox
    cdef rtype_t val

    cdef rtype_t *pinput, *pkinput, *poutput
    cdef rtype_t *pkernel, *pkernel_cur

    if not np.PyArray_ISCONTIGUOUS(kernel):
        kernel = kernel.copy()

    pkernel = <rtype_t*> kernel.data

    if output is None:
        output = np.empty((ymax, xmax), dtype=rtype)
    
    assert (output.shape[0] == ymax and output.shape[1] == xmax)

    sy = input.strides[0] / sizeof(rtype_t)
    sx = input.strides[1] / sizeof(rtype_t)
    soy = output.strides[0] / sizeof(rtype_t)
    sox = output.strides[1] / sizeof(rtype_t)

    pinput  = <rtype_t*> input.data
    poutput = <rtype_t*> output.data
    
    for y in range(ymax):
        for x in range(xmax):
            val = 0.

            pkinput = pinput
            pkernel_cur = pkernel
            for ky in range(kymax):
                for kx in range(kxmax):
                    val += pkinput[0] * pkernel_cur[0]
                    pkernel_cur += 1
                    pkinput += sx
                pkinput += sy

            poutput[0] = val

            pinput  += sx
            poutput += sox
    pinput  += sy
    poutput += soy

    return output

@cython.boundscheck(False)
def correlate_3d_valid(np.ndarray[rtype_t, ndim=3] input  not None,
                       np.ndarray[rtype_t, ndim=3] kernel not None,
                       np.ndarray[rtype_t, ndim=3] output = None):

    cdef int kzmax = kernel.shape[0]
    cdef int kymax = kernel.shape[1]
    cdef int kxmax = kernel.shape[2]
    cdef int zmax = input.shape[0] + 1 - kzmax
    cdef int ymax = input.shape[1] + 1 - kymax
    cdef int xmax = input.shape[2] + 1 - kxmax
    
    cdef int z, y, x, kz, ky, kx
    cdef rtype_t val
    
    if output is None:
        output = np.empty((zmax, ymax, xmax), dtype=rtype)
    
    assert (output.shape[0] == zmax and output.shape[1] == ymax and \
            output.shape[2] == xmax)
    
    for z in range(zmax):
        for y in range(ymax):
            for x in range(xmax):
                val = 0.
            
                for kz in range(kzmax):
                    for ky in range(kymax):
                        for kx in range(kxmax):
                            val += input[z+kz,y+ky,x+kx] * kernel[kz, ky, kx]

                output[z, y, x] = val

    return output
