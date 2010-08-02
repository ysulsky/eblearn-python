# mode -*-python-*-

from eblearn.gofast.util    cimport *
from eblearn.gofast.vecmath cimport *

from eblearn.idx     import unfold
from eblearn.vecmath import mkextmk, m2kdotmk

import scipy.signal


import_array()

sig_correlate = None
try:
    from scipy.signal import correlate as sig_correlate
except ImportError:
    pass

def gen_correlate_scipy(input, kernel, output=None, accumulate=False):
    y = sig_correlate(input, kernel, 'valid')
    if output is None: output    = y
    elif accumulate:   output   += y
    else:              output[:] = y
    return output

def gen_correlate_noscipy(input, kernel, output=None, accumulate=False):
    out_shape = tuple(np.subtract(input.shape, kernel.shape) + 1)
    if output is None:
        output = np.zeros(out_shape, input.dtype)
    assert (out_shape == output.shape), "shapes don't match"
    uin = input
    for d, kd in enumerate(kernel.shape):
        uin = unfold(uin, d, kd, 1)
    m2kdotmk(uin, kernel, output, accumulate)
    return output

if sig_correlate is None:
    gen_correlate = gen_correlate_noscipy
else:
    gen_correlate = gen_correlate_scipy

def m1_correlate(np.ndarray input not None, np.ndarray kernel not None,
                 np.ndarray output=None, bint accumulate=False):
    cdef np.ndarray uinput, rr
    assert (input.ndim == 1 and kernel.ndim == 1),   "wrong dimensions"
    input  = cvt(input,  NPY_RTYPE, 0)
    kernel = cvt(kernel, NPY_RTYPE, 0)
    uinput = unfold(input, 0, kernel.shape[0], 1)
    if output is None:
        rr = output = PyArray_EMPTY(1, uinput.shape, NPY_RTYPE, 0)
    else:
        assert (output.ndim == 1 and
                output.shape[0] == uinput.shape[0]), "shapes don't match"
        rr = cvt(output, NPY_RTYPE, RESULTFLAGS)
    c_m2dotm1(uinput, kernel, rr, accumulate)
    return output

def m2_correlate(np.ndarray input not None, np.ndarray kernel not None,
                 np.ndarray output=None, bint accumulate=False):
    cdef np.ndarray uinput, rr
    cdef int kh, kw
    assert (input.ndim == 2 and kernel.ndim == 2),   "wrong dimensions"
    kh, kw = kernel.shape[0], kernel.shape[1]
    input  = cvt(input,  NPY_RTYPE, 0)
    kernel = cvt(kernel, NPY_RTYPE, 0)
    uinput = unfold(unfold(input, 0, kh, 1), 1, kw, 1)
    if output is None:
        rr = output = PyArray_EMPTY(2, uinput.shape, NPY_RTYPE, 0)
    else:
        assert (output.ndim == 2 and
                output.shape[0] == uinput.shape[0] and
                output.shape[1] == uinput.shape[1]), "shapes don't match"
        rr = cvt(output, NPY_RTYPE, RESULTFLAGS)
    c_m4dotm2(uinput, kernel, rr, accumulate)
    return output

def m3_correlate(np.ndarray input not None, np.ndarray kernel not None,
                 np.ndarray output=None, bint accumulate=False):
    cdef np.ndarray uinput, rr
    cdef int kd, kh, kw
    assert (input.ndim == 3 and kernel.ndim == 3),   "wrong dimensions"
    kd, kh, kw = kernel.shape[0], kernel.shape[1], kernel.shape[2]
    input  = cvt(input,  NPY_RTYPE, 0)
    kernel = cvt(kernel, NPY_RTYPE, 0)
    uinput = unfold(unfold(unfold(input, 0, kd, 1), 1, kh, 1), 2, kw, 1)
    if output is None:
        rr = output = PyArray_EMPTY(3, uinput.shape, NPY_RTYPE, 0)
    else:
        assert (output.ndim == 3 and
                output.shape[0] == uinput.shape[0] and
                output.shape[1] == uinput.shape[1] and
                output.shape[2] == uinput.shape[2]), "shapes don't match"
        rr = cvt(output, NPY_RTYPE, RESULTFLAGS)
    c_m6dotm3(uinput, kernel, rr, accumulate)
    return output

def correlate_for_dim(int n):
    if n == 1: return m1_correlate
    if n == 2: return m2_correlate
    if n == 3: return m3_correlate
    return gen_correlate

def correlate(np.ndarray input, np.ndarray kernel, np.ndarray output=None,
              bint accumulate=False):
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
    return None

def m1_correlate_table(np.ndarray[int, ndim=2] table not None,
                       np.ndarray inputs  not None,
                       np.ndarray kernels not None,
                       np.ndarray outputs not None):
    cdef int kw
    cdef np.ndarray uinputs
    cdef int t, i, k, j
    
    assert (inputs.ndim==2 and kernels.ndim==2 and
            outputs.ndim==2),                        "wrong dimensions"
    
    inputs  = cvt(inputs,  NPY_RTYPE, 0)
    kernels = cvt(kernels, NPY_RTYPE, 0)
    outputs = cvt(outputs, NPY_RTYPE, RESULTFLAGS)
    
    kw = kernels.shape[1]
    uinputs = unfold(inputs, 1, kw, 1)
    
    assert (outputs.shape[1] == uinputs.shape[1]),   "shapes don't match"

    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        c_m2dotm1(uinputs[i], kernels[k], outputs[j], True)
    
    return None

def m2_correlate_table(np.ndarray[int, ndim=2] table not None,
                       np.ndarray inputs  not None,
                       np.ndarray kernels not None,
                       np.ndarray outputs not None):
    cdef int kh, kw
    cdef np.ndarray uinputs
    cdef int t, i, k, j
    
    assert (inputs.ndim==3 and kernels.ndim==3 and
            outputs.ndim==3),                        "wrong dimensions"
    
    inputs  = cvt(inputs,  NPY_RTYPE, 0)
    kernels = cvt(kernels, NPY_RTYPE, 0)
    outputs = cvt(outputs, NPY_RTYPE, RESULTFLAGS)
    
    kh, kw = kernels.shape[1], kernels.shape[2]
    uinputs = unfold(unfold(inputs, 1, kh, 1), 2, kw, 1)
    
    assert (outputs.shape[1] == uinputs.shape[1] and
            outputs.shape[2] == uinputs.shape[2]),   "shapes don't match"
    
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        c_m4dotm2(uinputs[i], kernels[k], outputs[j], True)
    
    return None

def m3_correlate_table(np.ndarray[int, ndim=2] table not None,
                       np.ndarray inputs  not None,
                       np.ndarray kernels not None,
                       np.ndarray outputs not None):
    cdef int kd, kh, kw
    cdef np.ndarray uinputs
    cdef int t, i, k, j
    
    assert (inputs.ndim==4 and kernels.ndim==4 and
            outputs.ndim==4),                        "wrong dimensions"
    
    inputs  = cvt(inputs,  NPY_RTYPE, 0)
    kernels = cvt(kernels, NPY_RTYPE, 0)
    outputs = cvt(outputs, NPY_RTYPE, RESULTFLAGS)
    
    kd, kh, kw = kernels.shape[1], kernels.shape[2], kernels.shape[3]
    uinputs = unfold(unfold(unfold(inputs, 1, kd, 1), 2, kh, 1), 3, kw, 1)
    
    assert (outputs.shape[1] == uinputs.shape[1] and
            outputs.shape[2] == uinputs.shape[2] and
            outputs.shape[3] == uinputs.shape[3]),   "shapes don't match"
    
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        c_m6dotm3(uinputs[i], kernels[k], outputs[j], True)

    return None

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
                   bint accumulate=False):
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
    return None

def m1_back_correlate_table(np.ndarray[int, ndim=2] table not None,
                            np.ndarray inputs  not None,
                            np.ndarray kernels not None,
                            np.ndarray outputs not None):
    cdef int kw
    cdef np.ndarray uoutputs
    cdef int t, i, k, j
    
    assert (inputs.ndim==2 and kernels.ndim==2 and
            outputs.ndim==2),                        "wrong dimensions"
    
    inputs  = cvt(inputs,  NPY_RTYPE, 0)
    kernels = cvt(kernels, NPY_RTYPE, 0)
    outputs = cvt(outputs, NPY_RTYPE, RESULTFLAGS)
    
    kw = kernels.shape[1]
    uoutputs = unfold(outputs, 1, kw, 1)
    
    assert (uoutputs.shape[1] == inputs.shape[1]  and
            uoutputs.shape[2] == kernels.shape[1]),  "shapes don't match"
    
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        c_m1extm1(inputs[i], kernels[k], uoutputs[j], True)
    
    return None

def m2_back_correlate_table(np.ndarray[int, ndim=2] table not None,
                            np.ndarray inputs  not None,
                            np.ndarray kernels not None,
                            np.ndarray outputs not None):
    cdef int kh, kw
    cdef np.ndarray uoutputs
    cdef int t, i, k, j
    
    assert (inputs.ndim==3 and kernels.ndim==3 and
            outputs.ndim==3),                        "wrong dimensions"
    
    inputs  = cvt(inputs,  NPY_RTYPE, 0)
    kernels = cvt(kernels, NPY_RTYPE, 0)
    outputs = cvt(outputs, NPY_RTYPE, RESULTFLAGS)
    
    kh, kw = kernels.shape[1], kernels.shape[2]
    uoutputs = unfold(unfold(outputs, 1, kh, 1), 2, kw, 1)
    
    assert (uoutputs.shape[1] == inputs.shape[1]  and
            uoutputs.shape[2] == inputs.shape[2]  and
            uoutputs.shape[3] == kernels.shape[1] and
            uoutputs.shape[4] == kernels.shape[2]),  "shapes don't match"
    
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        c_m2extm2(inputs[i], kernels[k], uoutputs[j], True)
    
    return None

def m3_back_correlate_table(np.ndarray[int, ndim=2] table not None,
                            np.ndarray inputs  not None,
                            np.ndarray kernels not None,
                            np.ndarray outputs not None):
    cdef int kd, kh, kw
    cdef np.ndarray uoutputs
    cdef int t, i, k, j
    
    assert (inputs.ndim==4 and kernels.ndim==4 and
            outputs.ndim==4),                        "wrong dimensions"
    
    inputs  = cvt(inputs,  NPY_RTYPE, 0)
    kernels = cvt(kernels, NPY_RTYPE, 0)
    outputs = cvt(outputs, NPY_RTYPE, RESULTFLAGS)
    
    kd, kh, kw = kernels.shape[1], kernels.shape[2], kernels.shape[3]
    uoutputs = unfold(unfold(unfold(outputs, 1, kd, 1), 2, kh, 1), 3, kw, 1)
    
    assert (uoutputs.shape[1] == inputs.shape[1]  and
            uoutputs.shape[2] == inputs.shape[2]  and
            uoutputs.shape[3] == inputs.shape[3]  and
            uoutputs.shape[4] == kernels.shape[1] and
            uoutputs.shape[5] == kernels.shape[2] and
            uoutputs.shape[6] == kernels.shape[3]),  "shapes don't match"
    
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        c_m3extm3(inputs[i], kernels[k], uoutputs[j], True)

    return None

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
