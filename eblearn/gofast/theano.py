from __future__ import absolute_import

from eblearn.util import rtype
from eblearn.idx  import reverse, reverse_along

from theano import function
from theano.tensor.signal.conv import conv2d
from theano.tensor import matrix, tensor3

import numpy as np

# todo: config_*

def make_theano_m2_convolve(mattype_inp, mattype_ker):
    dtype = np.dtype(rtype).name
    input  = mattype_inp('input',  dtype)
    kernel = mattype_ker('kernel', dtype)
    return function([input, kernel], conv2d(input, kernel))

theano_m2_convolve           = make_theano_m2_convolve(matrix,  matrix)
theano_m2_convolve_fulltable = make_theano_m2_convolve(tensor3, tensor3)

def m2_convolve(input, kernel, output=None, accumulate=False):
    res = theano_m2_convolve(input, kernel)[0]
    if output is None: return res
    if accumulate: output   += res
    else:          output[:] = res
    return output

def m2_convolve_fulltable(table, inputs, kernels, outputs):
    res = theano_m2_convolve_fulltable(inputs, kernels)
    for (i,k,j) in table: outputs[j] += res[i, k]
    return None

def m2_convolve_table(table, inputs, kernels, outputs):
    fn = theano_m2_convolve
    if len(table) == len(inputs)*len(kernels):
        return m2_convolve_fulltable(table, inputs, kernels, outputs)
    for (i,k,j) in table:
        outputs[j] += fn(inputs[i], kernels[k])[0]
    return None

def m2_correlate(input, kernel, output=None, accumulate=False):
    return m2_convolve(input, reverse(kernel), output, accumulate)

def m2_correlate_table(table, inputs, kernels, outputs):
    rev_kernels = reverse_along(reverse(kernels),0)
    return m2_convolve_table(table, inputs, rev_kernels, outputs)

