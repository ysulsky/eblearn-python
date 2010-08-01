# mode -*-python-*-

cimport cython

from _util cimport *
from _util import *

from _vecmath cimport *
from _vecmath import *

import scipy.signal
from _idx import *

import_array()

sig_correlate = scipy.signal.correlate
def gen_correlate(np.ndarray input, np.ndarray kernel, np.ndarray output=None,
                  bool accumulate=False):
    y = sig_correlate(input, kernel, 'valid')
    if output is None: output    = y
    elif accumulate:   output   += y
    else:              output[:] = y
    return output

def m1_correlate(np.ndarray[rtype_t, ndim=1] input  not None,
                 np.ndarray[rtype_t, ndim=1] kernel not None,
                 np.ndarray[rtype_t, ndim=1] output=None,
                 bool accumulate=False):
    cdef int kw = kernel.shape[0]
    cdef np.ndarray[rtype_t, ndim=2] uinput = unfold(input, 0, kw, 1)
    return c_m2dotm1(uinput, kernel, output, accumulate)

def m2_correlate(np.ndarray[rtype_t, ndim=2] input  not None,
                 np.ndarray[rtype_t, ndim=2] kernel not None,
                 np.ndarray[rtype_t, ndim=2] output=None,
                 bool accumulate=False):
    cdef int kh, kw
    cdef np.ndarray[rtype_t, ndim=4] uinput
    kh, kw = kernel.shape[0], kernel.shape[1]
    uinput = unfold(unfold(input, 0, kh, 1), 1, kw, 1)
    return c_m4dotm2(uinput, kernel, output, accumulate)

def m3_correlate(np.ndarray[rtype_t, ndim=3] input  not None,
                 np.ndarray[rtype_t, ndim=3] kernel not None,
                 np.ndarray[rtype_t, ndim=3] output=None,
                 bool accumulate=False):
    cdef int kd, kh, kw
    cdef np.ndarray[rtype_t, ndim=6] uinput
    kd, kh, kw = kernel.shape[0], kernel.shape[1], kernel.shape[2]
    uinput = unfold(unfold(unfold(input, 0, kd, 1), 1, kh, 1), 2, kw, 1)
    return c_m6dotm3(uinput, kernel, output, accumulate)

def correlate_for_dim(int n):
    if n == 1: return m1_correlate
    if n == 2: return m2_correlate
    if n == 3: return m3_correlate
    return gen_correlate

def correlate(np.ndarray input, np.ndarray kernel, np.ndarray output=None,
              bool accumulate=False):
    corrfn = correlate_for_dim(input.ndim)
    return corrfn(input, kernel, output, accumulate)

def gen_correlate_table(np.ndarray[int, ndim=2] table not None,
                        np.ndarray inputs             not None,
                        np.ndarray kernels            not None,
                        np.ndarray outputs            not None):
    cdef int t, i, k, j
    corrfn = correlate_for_dim(inputs.ndim-1)
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        corrfn(inputs[i], kernels[k], outputs[j], True)


def m1_correlate_table(np.ndarray[int,     ndim=2] table   not None,
                       np.ndarray[rtype_t, ndim=2] inputs  not None,
                       np.ndarray[rtype_t, ndim=2] kernels not None,
                       np.ndarray[rtype_t, ndim=2] outputs not None):
    cdef int kw
    cdef np.ndarray[rtype_t, ndim=3] uinputs
    cdef int t, i, k, j
    
    kh, kw = kernels.shape[1], kernels.shape[2]
    uinputs = unfold(inputs, 1, kw, 1)
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        c_m2dotm1(uinputs[i], kernels[k], outputs[j], True)


def m2_correlate_table(np.ndarray[int,     ndim=2] table   not None,
                       np.ndarray[rtype_t, ndim=3] inputs  not None,
                       np.ndarray[rtype_t, ndim=3] kernels not None,
                       np.ndarray[rtype_t, ndim=3] outputs not None):
    cdef int kh, kw
    cdef np.ndarray[rtype_t, ndim=5] uinputs
    cdef int t, i, k, j

    kh, kw = kernels.shape[1], kernels.shape[2]
    uinputs = unfold(unfold(inputs, 1, kh, 1), 2, kw, 1)
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        c_m4dotm2(uinputs[i], kernels[k], outputs[j], True)


def m3_correlate_table(np.ndarray[int,     ndim=2] table   not None,
                       np.ndarray[rtype_t, ndim=4] inputs  not None,
                       np.ndarray[rtype_t, ndim=4] kernels not None,
                       np.ndarray[rtype_t, ndim=4] outputs not None):
    cdef int kd, kh, kw
    cdef np.ndarray[rtype_t, ndim=7] uinputs
    cdef int t, i, k, j

    kd, kh, kw = kernels.shape[1], kernels.shape[2], kernels.shape[3]
    uinputs = unfold(unfold(unfold(inputs, 1, kd, 1), 2, kh, 1), 3, kw, 1)
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        c_m6dotm3(uinputs[i], kernels[k], outputs[j], True)


def correlate_table_for_dim(int n):
    if n == 1: return m1_correlate_table
    if n == 2: return m2_correlate_table
    if n == 3: return m3_correlate_table
    return gen_correlate_table

def correlate_table(np.ndarray[int, ndim=2] table not None,
                    np.ndarray inputs             not None,
                    np.ndarray kernels            not None,
                    np.ndarray outputs            not None):
    corrfn = correlate_table_for_dim(input.ndim-1)
    corrfn(table, inputs, kernels, outputs)

def back_correlate(np.ndarray input, np.ndarray kernel, np.ndarray output=None,
                   bool accumulate=False):
    cdef int d
    out_shape = tuple(np.subtract(np.shape(input), 1) + np.shape(kernel))
    if output is None:
        output = np.zeros(out_shape, input.dtype)
    assert (out_shape == np.shape(output)), "shapes don't match"
    uout = output
    for d in range(kernel.ndim):
        uout = unfold(uout, d, kernel[d], 1)
    mkextmk(input, kernel, uout, accumulate)
    return output

def back_correlate_for_dim(int n):
    # TODO
    return back_correlate

def gen_back_correlate_table(np.ndarray[int, ndim=2] table not None,
                             np.ndarray inputs             not None,
                             np.ndarray kernels            not None,
                             np.ndarray outputs            not None):
    cdef int t, i, k, j
    corrfn = back_correlate_for_dim(inputs.ndim-1)
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        corrfn(inputs[i], kernels[k], outputs[j], True)

def m1_back_correlate_table(np.ndarray[int, ndim=2] table       not None,
                            np.ndarray[rtype_t, ndim=2] inputs  not None,
                            np.ndarray[rtype_t, ndim=2] kernels not None,
                            np.ndarray[rtype_t, ndim=2] outputs not None):
    cdef int kw
    cdef np.ndarray[rtype_t, ndim=3] uoutputs
    cdef int t, i, k, j
    
    kw = kernels.shape[1]
    uoutputs = unfold(outputs, 1, kw, 1)
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        c_m1extm1(inputs[i], kernels[k], uoutputs[j], True)

def m2_back_correlate_table(np.ndarray[int, ndim=2] table       not None,
                            np.ndarray[rtype_t, ndim=3] inputs  not None,
                            np.ndarray[rtype_t, ndim=3] kernels not None,
                            np.ndarray[rtype_t, ndim=3] outputs not None):
    cdef int kh, kw
    cdef np.ndarray[rtype_t, ndim=5] uoutputs
    cdef int t, i, k, j
    
    kh, kw = kernels.shape[1], kernels.shape[2]
    uoutputs = unfold(unfold(outputs, 1, kh, 1), 2, kw, 1)
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        c_m2extm2(inputs[i], kernels[k], uoutputs[j], True)

def m3_back_correlate_table(np.ndarray[int, ndim=2] table       not None,
                            np.ndarray[rtype_t, ndim=4] inputs  not None,
                            np.ndarray[rtype_t, ndim=4] kernels not None,
                            np.ndarray[rtype_t, ndim=4] outputs not None):
    cdef int kd, kh, kw
    cdef np.ndarray[rtype_t, ndim=7] uoutputs
    cdef int t, i, k, j
    
    kd, kh, kw = kernels.shape[1], kernels.shape[2], kernels.shape[3]
    uoutputs = unfold(unfold(unfold(outputs, 1, kd, 1), 2, kh, 1), 3, kw, 1)
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        c_m3extm3(inputs[i], kernels[k], uoutputs[j], True)

def back_correlate_table_for_dim(int n):
    if n == 1: return m1_back_correlate_table
    if n == 2: return m2_back_correlate_table
    if n == 3: return m3_back_correlate_table
    return gen_back_correlate_table


def back_correlate_table(np.ndarray[int, ndim=2] table not None,
                         np.ndarray inputs             not None,
                         np.ndarray kernels            not None,
                         np.ndarray outputs            not None):
    corrfn = back_correlate_table_for_dim(input.ndim-1)
    corrfn(table, inputs, kernels, outputs)

