#!/usr/bin/env python

from eblearn      import vecmath
from eblearn.util import random, rtype

from math import sqrt
from timeit import Timer

import numpy as np

dtype = rtype
global_speed_test = 0
def cmp_slow_fast(name, *arg_dims, **kwargs):
    note      = kwargs.get('note', '')
    speedtest = max(global_speed_test, kwargs.get('speedtest',0))
    
    slow_fn = getattr(vecmath.slow_ver, name)
    fast_fn = getattr(vecmath.fast_ver, name, slow_fn)
    
    if slow_fn is fast_fn:
        print '*** Slow and fast versions of %s are the same' % (name,)
        return 
    
    def chg_scalars(args):
        for i in range(len(args)):
            if args[i].ndim == 0:
                args[i] = args[i].item()

    def speedup(mintime, fn1, args1, fn2, args2):
        i1 = i2 = 0
        n = 10
        t1 = Timer(lambda: fn1(*args1))
        t2 = Timer(lambda: fn2(*args2))
        while (i1 < mintime and i2 < mintime) or i2 < 1e-6:
            i1 = t1.timeit(n)
            i2 = t2.timeit(n)
            n *= 2
        warn = '' if i1 >= i2 else '  *** slowdown ***'
        si1 = '%.6g' % (i1,)
        si2 = '%.6g' % (i2,)
        return "%10s -> %-10s (%.3gx)%s" % (si1, si2, float(i1)/i2, warn)
    
    args1 = [random(dims).astype(dtype)*10-2 for dims in arg_dims]
    args2 = [arg1.copy() for arg1 in args1]
    
    args1_noncontig = [arg1.transpose().copy().transpose() for arg1 in args1]
    args2_noncontig = [arg2.transpose().copy().transpose() for arg2 in args2]

    chg_scalars(args1);           chg_scalars(args2)
    chg_scalars(args1_noncontig); chg_scalars(args2_noncontig)        
    
    ctg_res1 = slow_fn(*args1)
    ctg_res2 = fast_fn(*args2)
    ctg_err  = sqrt(np.sum(np.square(np.subtract(ctg_res1, ctg_res2))))

    nctg_res1 = slow_fn(*args1_noncontig)
    nctg_res2 = fast_fn(*args2_noncontig)
    nctg_err = sqrt(np.sum(np.square(np.subtract(nctg_res1, nctg_res2))))

    if note: note = ', '+note

    cmsg = 'SLOW vs. FAST (contiguous%s)    : %s' % (note, name)
    nmsg = 'SLOW vs. FAST (non-contiguous%s): %s' % (note, name)

    cspd  = nspd  = ''
    cpass = npass = 'FAIL'
    
    if ctg_err < 1e-6:
        cpass = 'pass'
        if speedtest:
            su = speedup(speedtest, slow_fn, args1, fast_fn, args2)
            cspd = '\n%s: %s' % ('...speed test ', su)
    
    if nctg_err < 1e-6:
        npass = 'pass'
        if speedtest:
            su = speedup(speedtest,
                         slow_fn, args1_noncontig,
                         fast_fn, args2_noncontig)
            nspd = '\n%s: %s' % ('...speed test ', su)
    
    print '%-45s error = %-15g %10s%s' % (cmsg, ctg_err,  cpass, cspd)
    print '%-45s error = %-15g %10s%s' % (nmsg, nctg_err, npass, nspd)

def test_slow_fast():
    if not vecmath.fast_ver:
        print "*** Can't find fast vecmath versions"
        return
    
    cmp_slow_fast('sumabs',  (3,4,5))
    cmp_slow_fast('sumsq',   (3,4,5))
    cmp_slow_fast('sqdist',  (3,4,5), (3,4,5))
    cmp_slow_fast('dtanh',   (30,20,20))
    cmp_slow_fast('ddtanh',  (30,20,20))
    cmp_slow_fast('m2dotm1', (30,400), (400,), (30,))
    cmp_slow_fast('m4dotm2', (3,4,5,6), (5,6), (3,4))
    cmp_slow_fast('m6dotm3', (3,4,5,6,7,8), (6,7,8), (3,4,5))
    cmp_slow_fast('m2kdotmk', (3,4,5,1,1,2, 6,7,8,1,2,1), (6,7,8,1,2,1),
                              (3,4,5,1,1,2))
    cmp_slow_fast('m1ldot',  (5,), (5,))
    cmp_slow_fast('m2ldot',  (30,50), (30,50))
    cmp_slow_fast('m3ldot',  (3,4,5), (3,4,5))
    cmp_slow_fast('ldot',    (30,2,5,2), (30,2,5,2))
    cmp_slow_fast('m1extm1', (30,), (40,), (30,40))
    cmp_slow_fast('m2extm2', (3,4), (5,6), (3,4,5,6))
    cmp_slow_fast('m3extm3', (3,4,5), (6,7,8), (3,4,5,6,7,8))
    cmp_slow_fast('mkextmk', (3,1,2,1,1,2), (6,3,2,1,2,1),
                             (3,1,2,1,1,2,   6,3,2,1,2,1))
    cmp_slow_fast('m2dotrows', (200,300), (200,300),)
    cmp_slow_fast('copy_normrows', (256,3,3))
    cmp_slow_fast('mdotc', (200,300), (), (200,300))

def test():
    test_slow_fast()
 
if __name__ == '__main__':
    #global_speed_test=0.5
    test()
