# mode -*-python-*-

from eblearn.gofast.util cimport *
from eblearn.idx          import narrow, reverse, reverse_along

import_array()

cdef extern from "ipp.h":
    void  ippStaticInit()

cdef extern from "ippi.h":
    ctypedef struct IppiSize:
        int width, height
    
    char* ippGetStatusString(int code)
    int   ippiConvValid_32f_C1R(float*, int, IppiSize,
                                float*, int, IppiSize, float*, int)

cdef str ipp_status_string(int code):
    return PyString_FromString(ippGetStatusString(code))

cdef int c_ipp_fconv2(np.ndarray input, np.ndarray kernel, np.ndarray output):
    cdef IppiSize input_size, kernel_size
    input_size.width  = input.shape[1];  input_size.height  = input.shape[0]
    kernel_size.width = kernel.shape[1]; kernel_size.height = kernel.shape[0]
    return ippiConvValid_32f_C1R\
        (<float*>input.data,  input.strides[0],  input_size,
         <float*>kernel.data, kernel.strides[0], kernel_size,
         <float*>output.data, output.strides[0])

def m2_convolve(np.ndarray input not None, np.ndarray kernel not None, 
                np.ndarray output = None, bint accumulate=False):
    cdef np.ndarray rr
    cdef int oh, ow, ret
    
    assert (input.ndim==2 and kernel.ndim==2), "wrong dimensions"
    oh = input.shape[0] - kernel.shape[0] + 1
    ow = input.shape[1] - kernel.shape[1] + 1
    
    if output is None:
        rr = output = PyArray_EMPTY2(oh, ow, np.NPY_FLOAT)
    else:
        assert (output.ndim     == 2  and 
                output.shape[0] == oh and
                output.shape[1] == ow), "shapes don't match"
        if accumulate:
            rr = PyArray_EMPTY2(oh, ow, np.NPY_FLOAT)
        else:
            rr = cvt(output, np.NPY_FLOAT, NPY_C_CONTIGUOUS | RESULTFLAGS)
    
    input  = cvt(input,  np.NPY_FLOAT, NPY_C_CONTIGUOUS)
    kernel = cvt(kernel, np.NPY_FLOAT, NPY_C_CONTIGUOUS)
    
    ret = c_ipp_fconv2(input, kernel, rr)
    if ret != 0: raise ValueError(ipp_status_string(ret))
    
    if accumulate: output += rr
    return output

def m2_convolve_table(np.ndarray[int, ndim=2] table not None,
                      np.ndarray inputs  not None,
                      np.ndarray kernels not None,
                      np.ndarray outputs not None):
    cdef int oh, ow, ret
    cdef int t, i, k, j
    cdef np.ndarray buf
    
    assert (inputs.ndim==3 and kernels.ndim==3 and
            outputs.ndim==3),                        "wrong dimensions"
    
    oh = inputs.shape[1] - kernels.shape[1] + 1
    ow = inputs.shape[2] - kernels.shape[2] + 1
    
    assert (outputs.shape[1] == oh and 
            outputs.shape[2] == ow),                 "shapes don't match"
    
    inputs  = cvt(inputs,  np.NPY_FLOAT, NPY_C_CONTIGUOUS)
    kernels = cvt(kernels, np.NPY_FLOAT, NPY_C_CONTIGUOUS)
    
    buf = PyArray_EMPTY2(oh, ow, np.NPY_FLOAT)
    
    for t in range(table.shape[0]):
        i = table[t,0]
        k = table[t,1]
        j = table[t,2]
        ret = c_ipp_fconv2(inputs[i], kernels[k], buf)
        if ret != 0:
            raise ValueError(ipp_status_string(ret))
        outputs[j] += buf
    
    return None

def m2_correlate(np.ndarray input not None, np.ndarray kernel not None, 
                 np.ndarray output = None, bint accumulate = False):
    return m2_convolve(input, reverse(kernel), output, accumulate)

def m2_correlate_table(np.ndarray[int, ndim=2] table not None,
                       np.ndarray inputs  not None,
                       np.ndarray kernels not None,
                       np.ndarray outputs not None):
    kernels = reverse_along(reverse(kernels), 0)
    m2_convolve_table(table, inputs, kernels, outputs)

ippStaticInit()

