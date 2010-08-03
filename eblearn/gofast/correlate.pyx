# mode -*-python-*-

from eblearn.gofast.util    cimport *
from eblearn.gofast.vecmath cimport *

from eblearn.idx     import reverse, reverse_along, unfold
from eblearn.vecmath import clear, mkextmk, m2kdotmk

import_array()

# TODO: write m[1-3]_back_{correlate,convolve}

cdef object vtbl_config_convolve  = None
cdef object vtbl_config_correlate = None
cdef object vtbl_config_back_convolve  = None
cdef object vtbl_config_back_correlate = None
def set_correlate_module_vtable(vtbl):
    global vtbl_config_convolve
    global vtbl_config_correlate
    global vtbl_config_back_convolve
    global vtbl_config_back_correlate
    vtbl_config_convolve  = vtbl['config_convolve']
    vtbl_config_correlate = vtbl['config_correlate']
    vtbl_config_back_convolve  = vtbl['config_back_convolve']
    vtbl_config_back_correlate = vtbl['config_back_correlate']


def m1_correlate(np.ndarray input not None, np.ndarray kernel not None,
                 np.ndarray output=None, bint accumulate=False):
    cdef np.ndarray uinput, rr
    assert (input.ndim == 1 and kernel.ndim == 1),   "wrong dimensions"
    input  = cvt(input,  NPY_RTYPE, 0)
    kernel = cvt(kernel, NPY_RTYPE, 0)
    uinput = unfold(input, 0, kernel.shape[0], 1)
    if output is None:
        rr = output = PyArray_ZEROS(1, uinput.shape, NPY_RTYPE, 0)
    else:
        assert (output.ndim == 1 and
                output.shape[0] == uinput.shape[0]), "shapes don't match"
        rr = cvt(output, NPY_RTYPE, RESULTFLAGS)
        if not accumulate: clear(rr)
    c_m2dotm1(uinput, kernel, rr, True)
    return output

cdef object _m1_correlate = m1_correlate
def m1_convolve(input, kernel, output=None, accumulate=False):
    return _m1_correlate(input, reverse(kernel), output, accumulate)


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
        rr = output = PyArray_ZEROS(2, uinput.shape, NPY_RTYPE, 0)
    else:
        assert (output.ndim == 2 and
                output.shape[0] == uinput.shape[0] and
                output.shape[1] == uinput.shape[1]), "shapes don't match"
        rr = cvt(output, NPY_RTYPE, RESULTFLAGS)
        if not accumulate: clear(rr)
    c_m4dotm2(uinput, kernel, rr, True)
    return output

cdef object _m2_correlate = m2_correlate
def m2_convolve(input, kernel, output=None, accumulate=False):
    return _m2_correlate(input, reverse(kernel), output, accumulate)


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
        rr = output = PyArray_ZEROS(3, uinput.shape, NPY_RTYPE, 0)
    else:
        assert (output.ndim == 3 and
                output.shape[0] == uinput.shape[0] and
                output.shape[1] == uinput.shape[1] and
                output.shape[2] == uinput.shape[2]), "shapes don't match"
        rr = cvt(output, NPY_RTYPE, RESULTFLAGS)
        if not accumulate: clear(rr)
    c_m6dotm3(uinput, kernel, rr, True)
    return output

cdef object _m3_correlate = m3_correlate
def m3_convolve(input, kernel, output=None, accumulate=False):
    return _m3_correlate(input, reverse(kernel), output, accumulate)


def gen_correlate_table(np.ndarray[int, ndim=2] table not None,
                        inputs, kernels, outputs):
    cdef int t, i, k, j
    corrfn = vtbl_config_correlate(inputs.ndim - 1, 
                                   inputs.shape[1:], kernels.shape[1:])
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        corrfn(inputs[i], kernels[k], outputs[j], True)
    return None

def gen_convolve_table (np.ndarray[int, ndim=2] table not None,
                        inputs, kernels, outputs):
    cdef int t, i, k, j
    convfn = vtbl_config_convolve (inputs.ndim - 1, 
                                   inputs.shape[1:], kernels.shape[1:])
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        convfn(inputs[i], kernels[k], outputs[j], True)
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

cdef object _m1_correlate_table = m1_correlate_table
def m1_convolve_table(table, inputs, kernels, outputs):
    rev_kernels = reverse_along(reverse(kernels), 0)
    _m1_correlate_table(table, inputs, rev_kernels, outputs)

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

cdef object _m2_correlate_table = m2_correlate_table
def m2_convolve_table(table, inputs, kernels, outputs):
    rev_kernels = reverse_along(reverse(kernels), 0)
    _m2_correlate_table(table, inputs, rev_kernels, outputs)

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

cdef object _m3_correlate_table = m3_correlate_table
def m3_convolve_table(table, inputs, kernels, outputs):
    rev_kernels = reverse_along(reverse(kernels), 0)
    _m3_correlate_table(table, inputs, rev_kernels, outputs)


def gen_back_correlate_table(np.ndarray[int, ndim=2] table not None,
                             inputs, kernels, outputs):
    cdef int t, i, k, j
    corrfn = vtbl_config_back_correlate(inputs.ndim - 1,
                                        inputs.shape[1:], kernels.shape[1:])
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        corrfn(inputs[i], kernels[k], outputs[j], True)
    return None


def gen_back_convolve_table (np.ndarray[int, ndim=2] table not None,
                        inputs, kernels, outputs):
    cdef int t, i, k, j
    convfn = vtbl_config_back_convolve (inputs.ndim - 1, 
                                        inputs.shape[1:], kernels.shape[1:])
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        convfn(inputs[i], kernels[k], outputs[j], True)
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

cdef object _m1_back_correlate_table = m1_back_correlate_table
def m1_back_convolve_table(table, inputs, kernels, outputs):
    rev_kernels = reverse_along(reverse(kernels), 0)
    _m1_back_correlate_table(table, inputs, rev_kernels, outputs)

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

cdef object _m2_back_correlate_table = m2_back_correlate_table
def m2_back_convolve_table(table, inputs, kernels, outputs):
    rev_kernels = reverse_along(reverse(kernels), 0)
    _m2_back_correlate_table(table, inputs, rev_kernels, outputs)

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

cdef object _m3_back_correlate_table = m3_back_correlate_table
def m3_back_convolve_table(table, inputs, kernels, outputs):
    rev_kernels = reverse_along(reverse(kernels), 0)
    _m3_back_correlate_table(table, inputs, rev_kernels, outputs)

