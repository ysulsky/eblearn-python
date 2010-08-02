from eblearn.idx     import unfold
from eblearn.vecmath import m2kdotmk, mkextmk

import numpy as np

try:
    import scipy.signal
    sig_correlate = scipy.signal.correlate
    def correlate(input, kernel, output=None, accumulate=False):
        y = sig_correlate(input, kernel, 'valid')
        if output is None: output    = y
        elif accumulate:   output   += y
        else:              output[:] = y
        return output
except ImportError:
    # no scipy
    def correlate(input, kernel, output=None, accumulate=False):
        out_shape = tuple(np.subtract(input.shape, kernel.shape) + 1)
        if output is None:
            output = np.zeros(out_shape, input.dtype)
        assert (out_shape == output.shape), "shapes don't match"
        uin = input
        for d, kd in enumerate(kernel.shape):
            uin = unfold(uin, d, kd, 1)
        m2kdotmk(uin, kernel, output, accumulate)
        return output

def correlate_table(table, inputs, kernels, outputs):
    for (i,k,j) in table:
        correlate(inputs[i], kernels[k], outputs[j], True)

correlate_for_dim       = lambda n: correlate
correlate_table_for_dim = lambda n: correlate_table

def back_correlate(input, kernel, output=None, accumulate=False):
    out_shape = tuple(np.subtract(input.shape, 1) + kernel.shape)
    if output is None:
        output = np.zeros(out_shape, input.dtype)
    assert (out_shape == output.shape), "shapes don't match"
    uout = output
    for d, kd in enumerate(kernel.shape):
        uout = unfold(uout, d, kd, 1)
    mkextmk(input, kernel, uout, accumulate)
    return output

def back_correlate_table(table, inputs, kernels, outputs):
    for (i,k,j) in table:
        back_correlate(inputs[i], kernels[k], outputs[j], True)

back_correlate_for_dim       = lambda n: back_correlate
back_correlate_table_for_dim = lambda n: back_correlate_table

__all__ = ['correlate',            'correlate_for_dim',
           'correlate_table',      'correlate_table_for_dim',
           'back_correlate',       'back_correlate_for_dim',
           'back_correlate_table', 'back_correlate_table_for_dim']

slow_ver = dict([(k, globals()[k]) for k in __all__])

try:
    from gofast.correlate import *
    have_fast = True
except ImportError:
    have_fast = False

fast_ver = dict([(k, globals()[k]) for k in __all__])

non_ipp_correlate_for_dim       = correlate_for_dim
non_ipp_correlate_table_for_dim = correlate_table_for_dim
try:
    from gofast.ipp import *
    
    def correlate_for_dim(n):
        if n == 2: return m2_correlate
        return non_ipp_correlate_for_dim(n)
    
    def correlate_table_for_dim(n):
        if n == 2: return m2_correlate_table
        return non_ipp_correlate_table_for_dim(n)
    
    have_ipp = True
except ImportError:
    have_ipp = False

ipp_ver = dict([(k, globals()[k]) for k in __all__])


def eblearn_disable_ipp():
    from eblearn.util import replace_global
    for k in fast_ver:
        ippfn, fastfn = ipp_ver[k], fast_ver[k]
        if ippfn is not fastfn:
            replace_global('eblearn', ippfn, fastfn)

def eblearn_enable_ipp():
    from eblearn.util import replace_global
    for k in fast_ver:
        ippfn, fastfn = ipp_ver[k], fast_ver[k]
        if ippfn is not fastfn:
            replace_global('eblearn', fastfn, ippfn)
        
