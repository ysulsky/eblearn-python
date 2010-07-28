import numpy as np
from math import sqrt
from time import clock

from eblearn        import vecmath as slow_vecmath
from eblearn.gofast import vecmath as fast_vecmath

def cmp_slow_fast(name, *arg_dims, **kwargs):
    note = kwargs.get('note', '')
    speedtest = kwargs.get('speedtest',0)
    
    def chg_scalars(args):
        for i in range(len(args)):
            if args[i].ndim == 0:
                args[i] = args[i].item()

    def speedup(n, fn1, args1, fn2, args2):
        t1 = clock()
        for i in xrange(n):
            fn1(*args1)
        t2 = clock()
        for i in xrange(n):
            fn2(*args2)
        t3 = clock()
        if t3 - t2 < 1e-8:
            return speedup(n * 10, fn1, args1, fn2, args2)
        return (t2 - t1) / (t3 - t2)
    
    args1 = [np.random.random(dims)*4-2 for dims in arg_dims]
    args2 = [arg1.copy() for arg1 in args1]

    args1_noncontig = [arg1.T.copy().T for arg1 in args1]
    args2_noncontig = [arg2.T.copy().T for arg2 in args2]

    chg_scalars(args1);           chg_scalars(args2)
    chg_scalars(args1_noncontig); chg_scalars(args2_noncontig)
    
    slow_fn = getattr(slow_vecmath, name)
    fast_fn = getattr(fast_vecmath, name)
    
    ctg_res1 = slow_fn(*args1)
    ctg_res2 = fast_fn(*args2)
    ctg_err  = sqrt(np.sum(np.square(np.subtract(ctg_res1, ctg_res2))))

    nctg_res1 = slow_fn(*args1_noncontig)
    nctg_res2 = fast_fn(*args2_noncontig)
    nctg_err = sqrt(np.sum(np.square(np.subtract(nctg_res1, nctg_res2))))

    if note: note = ', '+note

    cmsg = 'SLOW vs. FAST (contiguous%s)    : %s' % (note, name)
    nmsg = 'SLOW vs. FAST (non-contiguous%s): %s' % (note, name)

    cpass = npass = 'FAIL'
    if ctg_err < 1e-6:
        if speedtest:
            su = speedup(speedtest, slow_fn, args1, fast_fn, args2)
            cpass = 'pass (%fx)' % su
        else:
            cpass = 'pass'

    if nctg_err < 1e-6:
        if speedtest:
            su = speedup(speedtest,
                         slow_fn, args1_noncontig,
                         fast_fn, args2_noncontig)
            npass = 'pass (%fx)' % su
        else:
            npass = 'pass'
    
    print '%-40s error = %-15g %15s' % (cmsg, ctg_err,  cpass)
    print '%-40s error = %-15g %15s' % (nmsg, nctg_err, npass)

def test():
    fast_fns = fast_vecmath.__all__
    try:
        fast_vecmath.__all__ = []
        reload(slow_vecmath)

        cmp_slow_fast('sqmag',   (3,4,5) )
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
        cmp_slow_fast('m1extm1', (30,), (400,), (30,400))
        cmp_slow_fast('m2extm2', (3,4), (5,6), (3,4,5,6))
        cmp_slow_fast('m3extm3', (3,4,5), (6,7,8), (3,4,5,6,7,8))
        cmp_slow_fast('mkextmk', (3,1,2,1,1,2), (6,3,2,1,2,1),
                                 (3,1,2,1,1,2,   6,3,2,1,2,1))
        cmp_slow_fast('m2dotrows', (200,300), (200,300),)
        cmp_slow_fast('copy_normrows', (256,3,3), speedtest=100)
    
    finally:
        fast_vecmath.__all__ = fast_fns
        reload(slow_vecmath)

if __name__ == '__main__':
    test()
