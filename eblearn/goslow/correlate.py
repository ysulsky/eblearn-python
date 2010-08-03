from eblearn.idx     import unfold, reverse, reverse_along
from eblearn.vecmath import clear, m2kdotmk, mkextmk

import numpy as np

### MODULE VTABLE ##########################################

def set_correlate_module_vtable(vtbl):
    mod_globals = globals()
    for k,v in vtbl.iteritems(): 
        mod_globals['vtbl_' + k] = v

### CORRELATE & CONVOLVE ###################################

sig_correlate = None
sig_convolve  = None
try:
    from scipy.signal import \
        correlate as sig_correlate, convolve as sig_convolve
except ImportError:
    pass

def gen_correlate_scipy(input, kernel, output=None, accumulate=False):
    y = sig_correlate(input, kernel, 'valid')
    if output is None: output    = y
    elif accumulate:   output   += y
    else:              output[:] = y
    return output

def gen_convolve_scipy (input, kernel, output=None, accumulate=False):
    y = sig_convolve (input, kernel, 'valid')
    if output is None: output    = y
    elif accumulate:   output   += y
    else:              output[:] = y
    return output

def gen_correlate_noscipy(input, kernel, output=None, accumulate=False):
    out_shape = tuple(np.subtract(input.shape, kernel.shape) + 1)
    if output is None:
        output = np.zeros(out_shape, input.dtype)
    else:
        assert (out_shape == output.shape), "shapes don't match"
        if not accumulate: clear(output)
    uin = input
    for d, kd in enumerate(kernel.shape):
        uin = unfold(uin, d, kd, 1)
    m2kdotmk(uin, kernel, output, True)
    return output

def gen_convolve_noscipy (input, kernel, output=None, accumulate=False):
    return gen_correlate_noscipy(input, reverse(kernel), output, accumulate)

if sig_correlate is None:
    gen_correlate = gen_correlate_noscipy
    gen_convolve  = gen_convolve_noscipy
else:
    gen_correlate = gen_correlate_scipy
    gen_convolve  = gen_convolve_scipy

m1_correlate = gen_correlate; m1_convolve = gen_convolve
m2_correlate = gen_correlate; m2_convolve = gen_convolve
m3_correlate = gen_correlate; m3_convolve = gen_convolve

def config_correlate(ndim=-1, inshape=None, kernshape=None):
    if ndim == 1: return vtbl_m1_correlate
    if ndim == 2: return vtbl_m2_correlate
    if ndim == 3: return vtbl_m3_correlate
    return vtbl_gen_correlate

def config_convolve (ndim=-1, inshape=None, kernshape=None):
    if ndim == 1: return vtbl_m1_convolve
    if ndim == 2: return vtbl_m2_convolve
    if ndim == 3: return vtbl_m3_convolve
    return vtbl_gen_convolve

def correlate(input, kernel, output=None, accumulate=False):
    fn = vtbl_config_correlate(input.ndim, input.shape, kernel.shape)
    return fn(input, kernel, output, accumulate)

def convolve (input, kernel, output=None, accumulate=False):
    fn = vtbl_config_convolve (input.ndim, input.shape, kernel.shape)
    return fn(input, kernel, output, accumulate)

### BACKWARD CORRELATE & CONVOLVE ##########################

def gen_back_correlate(input, kernel, output=None, accumulate=False):
    out_shape = tuple(np.subtract(input.shape, 1) + kernel.shape)
    if output is None:
        output = np.zeros(out_shape, input.dtype)
    else:
        assert (out_shape == output.shape), "shapes don't match"
        if not accumulate: clear(output)
    uout = output
    for d, kd in enumerate(kernel.shape):
        uout = unfold(uout, d, kd, 1)
    mkextmk(input, kernel, uout, True)
    return output

def gen_back_convolve (input, kernel, output=None, accumulate=False):
    return gen_back_correlate(input, reverse(kernel), output, accumulate)

m1_back_correlate = gen_back_correlate; m1_back_convolve = gen_back_convolve
m2_back_correlate = gen_back_correlate; m2_back_convolve = gen_back_convolve
m3_back_correlate = gen_back_correlate; m3_back_convolve = gen_back_convolve

def config_back_correlate(ndim=-1, inshape=None, kernshape=None):
    if ndim == 1: return vtbl_m1_back_correlate
    if ndim == 2: return vtbl_m2_back_correlate
    if ndim == 3: return vtbl_m3_back_correlate
    return vtbl_gen_back_correlate

def config_back_convolve (ndim=-1, inshape=None, kernshape=None):
    if ndim == 1: return vtbl_m1_back_convolve
    if ndim == 2: return vtbl_m2_back_convolve
    if ndim == 3: return vtbl_m3_back_convolve
    return vtbl_gen_back_convolve

def back_correlate(input, kernel, output=None, accumulate=False):
    fn = vtbl_config_back_correlate(input.ndim, input.shape, kernel.shape)
    return fn(input, kernel, output, accumulate)

def back_convolve (input, kernel, output=None, accumulate=False):
    fn = vtbl_config_back_convolve (input.ndim, input.shape, kernel.shape)
    return fn(input, kernel, output, accumulate)

### CORRELATE & CONVOLVE TABLES ############################

def gen_correlate_table(table, inputs, kernels, outputs):
    fn = vtbl_config_correlate(inputs.ndim-1,
                               inputs.shape[1:], kernels.shape[1:])
    for (i,k,j) in table:
        fn(inputs[i], kernels[k], outputs[j], True)
    return None

def gen_convolve_table (table, inputs, kernels, outputs):
    fn = vtbl_config_convolve (inputs.ndim-1,
                              inputs.shape[1:], kernels.shape[1:])
    for (i,k,j) in table:
        fn(inputs[i], kernels[k], outputs[j], True)
    return None

m1_correlate_table = gen_correlate_table; m1_convolve_table = gen_convolve_table
m2_correlate_table = gen_correlate_table; m2_convolve_table = gen_convolve_table
m3_correlate_table = gen_correlate_table; m3_convolve_table = gen_convolve_table

def config_correlate_table(ndim    = -1,   table     = None,
                           inshape = None, kernshape = None):
    if ndim == 1: return vtbl_m1_correlate_table
    if ndim == 2: return vtbl_m2_correlate_table
    if ndim == 3: return vtbl_m3_correlate_table
    return vtbl_gen_correlate_table

def config_convolve_table (ndim    = -1,   table     = None,
                           inshape = None, kernshape = None):
    if ndim == 1: return vtbl_m1_convolve_table
    if ndim == 2: return vtbl_m2_convolve_table
    if ndim == 3: return vtbl_m3_convolve_table
    return vtbl_gen_convolve_table

def correlate_table(table, inputs, kernels, outputs):
    fn = vtbl_config_correlate_table(inputs.ndim-1, table,
                                     inputs.shape, kernels.shape)
    fn(table, inputs, kernels, outputs)

def convolve_table (table, inputs, kernels, outputs):
    fn = vtbl_config_convolve_table (inputs.ndim-1, table,
                                     inputs.shape, kernels.shape)
    fn(table, inputs, kernels, outputs)

### BACKWARD CORRELATE & CONVOLVE TABLES ###################

def gen_back_correlate_table(table, inputs, kernels, outputs):
    fn = vtbl_config_back_correlate(inputs.ndim-1,
                                    inputs.shape[1:], kernels.shape[1:])
    for (i,k,j) in table:
        fn(inputs[i], kernels[k], outputs[j], True)
    return None

def gen_back_convolve_table (table, inputs, kernels, outputs):
    fn = vtbl_config_back_convolve (inputs.ndim-1,
                                    inputs.shape[1:], kernels.shape[1:])
    for (i,k,j) in table:
        fn(inputs[i], kernels[k], outputs[j], True)
    return None

m1_back_correlate_table = gen_back_correlate_table
m1_back_convolve_table  = gen_back_convolve_table
m2_back_correlate_table = gen_back_correlate_table
m2_back_convolve_table  = gen_back_convolve_table
m3_back_correlate_table = gen_back_correlate_table
m3_back_convolve_table  = gen_back_convolve_table

def config_back_correlate_table(ndim=-1, table=None,
                                inshape=None, kernshape=None):
    if ndim == 1: return vtbl_m1_back_correlate_table
    if ndim == 2: return vtbl_m2_back_correlate_table
    if ndim == 3: return vtbl_m3_back_correlate_table
    return vtbl_gen_back_correlate_table

def config_back_convolve_table  (ndim=-1, table=None,
                                 inshape=None, kernshape=None):
    if ndim == 1: return vtbl_m1_back_convolve_table
    if ndim == 2: return vtbl_m2_back_convolve_table
    if ndim == 3: return vtbl_m3_back_convolve_table
    return vtbl_gen_back_convolve_table

def back_correlate_table(table, inputs, kernels, outputs):
    fn = vtbl_config_back_correlate_table(inputs.ndim-1, table,
                                          inputs.shape, kernels.shape)
    fn(table, inputs, kernels, outputs)

def back_convolve_table (table, inputs, kernels, outputs):
    fn = vtbl_config_back_convolve_table (inputs.ndim-1, table,
                                          inputs.shape, kernels.shape)
    fn(table, inputs, kernels, outputs)

