#!/usr/bin/python

# Ugly testing code. TODO: clean me up.
#
# Tests all combinations of [config_][back_]{correlate,convolve}[_table]
# - tests all implementations in eblearn.correlate
# - compares all results to those of the slow version
# - tests partial and full tables
# - tests a range of dimensions
# - tests functional/3-argument/accumulate modes for non-table versions

import eblearn.correlate
from eblearn.util    import debug_break, random, rtype
from eblearn.vecmath import clear, sqdist

from numpy   import asarray, zeros
from math    import sqrt
from timeit  import Timer

all_vers = eblearn.correlate.all_vers

test_inputs = [(10000,), (100, 200), (20,30,15), (7,8,9,10)]
test_kernels= [(50,),    (8,  10),   (5, 6, 4),  (2,3,4,5) ]
dim_range   = 1,4

test_table_in  = 2
test_table_out = 2

def _make_arg(size1, size_override, dtype):
    size = size1 if size_override is None else size_override
    assert ((size_override is None) or (len(size1) == len(size_override)))
    return random(size).astype(dtype)

make_input_single  = lambda d, s=None, t=rtype: _make_arg(test_inputs[d-1], s,t)
make_kernel_single = lambda d, s=None, t=rtype: _make_arg(test_kernels[d-1],s,t)
make_input_table  = lambda d, s=None, t=rtype: \
    asarray([make_input_single(d, s, t) 
             for i in range(test_table_in)])
make_kernel_table = lambda d, s=None, t=rtype: \
    asarray([make_kernel_single(d, s) 
             for i in range(test_table_in *test_table_out)])

test_conn_full = [(a,b)
                  for a in range(test_table_in) 
                  for b in range(test_table_out)]
test_full_table=asarray([(a,i,b) for
                         (i,(a,b)) in enumerate(test_conn_full)],'i')
test_part_table=asarray([x 
                         for (i,x) in enumerate(test_full_table) if i%2],'i')

def test_close(n1, v1, n2, v2):
    name = "|%s - %s|" % (n1, n2)
    dist = sqrt(sqdist(v1, v2))
    print '%50s = %-12g%10s' % (name, dist, "pass" if dist < 1e-3 else "FAIL")

def get_single_fn(f, input, kernel, table, configure):
    assert table is None
    cf = f if not configure else f(input.ndim, input.shape, kernel.shape)
    return lambda table, *args: cf(*args)

def test_modes_single(f, input, kernel, table = None, configure = False, fwd=None):
    assert table is None
    f = get_single_fn(f, input, kernel, table, configure)
    y1  = f(table, input, kernel)                # functional mode
    y2  = y1.copy().astype(input.dtype)
    clear(y2)
    y2_ = f(table, input, kernel, y2)            # copy-into mode
    assert (y2 is y2_)
    test_close("f(x,k)", y1, "f(x,k,y)", y2)
    y3  = y1 * 1.5
    f(table, input, kernel, y3, True)            # accumulate mode
    test_close("f(x,k,y*1.5,True)", y3, "f(x,k,y)*2.5", y1*2.5)
    return y1

def get_table_fn(f, inputs, kernels, table, configure):
    if not configure: return f
    return f(inputs.ndim-1, table, inputs.shape, kernels.shape)

def test_modes_table(f, inputs, kernels, table = None, configure = False, fwd=None):
    # there's just one mode for tables
    
    assert table is not None
    assert fwd is not None
    
    size_out = [max(table[:,2])+1]
    if fwd:
        size_out += [xd-kd+1 for xd,kd in zip(inputs.shape[1:],
                                              kernels.shape[1:])]
    else:
        size_out += [xd+kd-1 for xd,kd in zip(inputs.shape[1:],
                                              kernels.shape[1:])]
    
    outputs = zeros(size_out, dtype = inputs.dtype)
    f = get_table_fn(f, inputs, kernels, table, configure)
    f(table, inputs, kernels, outputs)
    return outputs
    
def reset_to_ver(target_ver):
    slow_ver = eblearn.correlate.slow_ver
    for name, ver in all_vers.items():
        enabled = ver in (slow_ver, target_ver)
        setattr(eblearn.correlate, "%s_enabled" % (name,), enabled)
        eblearn.correlate.reset_implementations(packages=None)

def save_enabled_vers():
    return [getattr(eblearn.correlate, "%s_enabled" % (name,)) 
            for name in all_vers if name != 'slow_ver']
    
def restore_enabled_vers(enabled):
    for prev_enabled, name in zip(enabled, all_vers):
        if name != 'slow_ver':
            setattr(eblearn.correlate, "%s_enabled" % (name,), prev_enabled)
    eblearn.correlate.reset_implementations(packages=None)


def test_version(test_ver_name, test_ver, use_tables = False):
    
    make_input   = make_input_table  if use_tables else make_input_single
    make_kernel  = make_kernel_table if use_tables else make_kernel_single
    configure_fn = get_table_fn if use_tables else get_single_fn
    test_modes   = test_modes_table if use_tables else test_modes_single
    tables       = [('full table', test_full_table),
                    ('part table', test_part_table)] \
                    if use_tables else [('no table', None)]
    
    assert test_ver_name in all_vers and all_vers[test_ver_name] is test_ver
    
    slow_ver = eblearn.correlate.slow_ver
    
    cur_enabled = save_enabled_vers()
    try:        
        for (d, (tabname, table), (confname, configured)) in \
                [(a,t,b)
                 for a in range(dim_range[0], dim_range[1]+1)
                 for t in tables
                 for b in (('not configured', False), ('configured', True))]:

            x = make_input(d)
            k = make_kernel(d)
            config_prefix = '' if not configured else 'config_'
            print '  -- %d dimensional inputs (%s, %s) -- ' % \
                (d, tabname, confname)
            for (func,fwd) in [(config_prefix + prefix + fn, prefix != 'back_')
                               for prefix in ('','back_')
                               for fn     in ('correlate', 'convolve')]:
           
                reset_to_ver(test_ver)
                
                if table is not None: func += '_table'
                fn = getattr(eblearn.correlate, func)
                print '  '+func
                if test_ver is not slow_ver or table is None:
                    y1 = test_modes(fn, x, k, table, configured, fwd)
                if test_ver is not slow_ver:
                    y2 = y1.copy(); clear(y2)
                    m2_correlate_fast = getattr(test_ver, 'm2_correlate', None)
                    assert ((m2_correlate_fast is None) or
                            (eblearn.correlate.m2_correlate is m2_correlate_fast))
                    reset_to_ver(slow_ver)
                    m2_correlate_slow = getattr(slow_ver, 'm2_correlate')
                    assert (eblearn.correlate.m2_correlate is m2_correlate_slow)
                    assert (m2_correlate_slow is not m2_correlate_fast)
                    oldfn = getattr(eblearn.correlate, func)
                    configure_fn(oldfn, x, k, table, configured)(table,x,k,y2)
                    test_close("f(x,k)", y1, "slow_f(x,k)", y2)
            print '  --------------------------- '
    
    finally:
        restore_enabled_vers(cur_enabled)

def speedtest(n, ver_name, fname, ndim,
              indims=None, kerdims=None, table=test_full_table, dtype=rtype):
    ''' e.g. speedtest(100, 'ipp_ver', 'back_correlate', 2) '''
    ver = getattr(eblearn.correlate, ver_name)
    if ver is None:
        print "*** %s isn't loaded" % (ver_name,)
        return None
    cur_enabled = save_enabled_vers()
    try:
        reset_to_ver(ver)
        is_table  = 'table'  in fname
        is_config = 'config' in fname
        is_back   = 'back'   in fname
        
        xs = make_input_single(ndim, indims, dtype)
        ks = make_kernel_single(ndim, kerdims, dtype)
        
        if is_back:
            ys = zeros(asarray(xs.shape)+ks.shape-1,dtype=dtype)
        else:
            ys = zeros(asarray(xs.shape)-ks.shape+1,dtype=dtype)
        
        if is_table:
            num_in  = max(table[:,0])+1
            num_ker = max(table[:,1])+1
            num_out = max(table[:,2])+1
            
            xs = asarray([xs for i in range(num_in)])
            ks = asarray([ks for i in range(num_ker)])
            ys = asarray([ys for i in range(num_out)])
            
            xs[:] = random(xs.shape)
            ks[:] = random(ks.shape)

        fn = getattr(eblearn.correlate, fname)
        if is_config:
            if is_table:
                fn = fn(ndim, table, xs.shape, ks.shape)
            else:
                fn = fn(ndim, xs.shape, ks.shape)
        
        if is_table:
            return Timer(lambda: fn(table, xs, ks, ys)).timeit(n)
        else:
            return Timer(lambda: fn(xs, ks, ys)).timeit(n)
    
    finally:
        restore_enabled_vers(cur_enabled)
    

def test():
    for ver_name,ver in all_vers.items():
        
        if ver is None:
            print " **** Version %s is not loaded. Not testing." % (ver_name,)
            continue
  
        print "###################################"
        print "# TESTING %s" % (ver_name,)
    
        test_version(ver_name, ver, False)
        test_version(ver_name, ver, True)


if __name__ == '__main__':
    test_version('theano_ver', all_vers['theano_ver'], False)
    test_version('theano_ver', all_vers['theano_ver'], 1)
    #test()


