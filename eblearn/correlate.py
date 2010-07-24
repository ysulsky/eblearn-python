import scipy.signal
from idx     import *
from vecmath import fast as fast_vecmath, m4dotm2, m6dotm3

sig_correlate = scipy.signal.correlate

def m1_correlate(input, kernel, output=None, accumulate=False):
    kw, = kernel.shape
    uinput = unfold(input, 0, kw, 1)
    return m2dotm1(uinput, kernel, output, accumulate)

def m2_correlate(input, kernel, output=None, accumulate=False):
    kh, kw = kernel.shape
    uinput = unfold(unfold(input, 0, kh, 1), 1, kw, 1)
    return m4dotm2(uinput, kernel, output, accumulate)

def m3_correlate(input, kernel, output=None, accumulate=False):
    kd, kh, kw = kernel.shape
    uinput = unfold(unfold(unfold(input, 0, kd, 1), 1, kh, 1), 2, kw, 1)
    return m6dotm3(uinput, kernel, output, accumulate)

def gen_correlate(input, kernel, output=None, accumulate=False):
    y = sig_correlate(input, kernel, 'valid')
    if output is None: output    = y
    elif accumulate:   output   += y
    else:              output[:] = y
    return output

if not fast_vecmath:
    correlate = gen_correlate
else:
    corr_funs = [gen_correlate, m1_correlate, m2_correlate, m3_correlate]
    def correlate(input, kernel, output=None, accumulate=False):
        if kernel.ndim < len(corr_funs):
            return corr_funs[kernel.ndim](input, kernel, output, accumulate)
        return gen_correlate(input, kernel, output, accumulate)

__all__ = ['correlate']
