from math import sqrt
import numpy as np

clear  = lambda x: x.fill(0)
sqmag  = lambda x: np.square(x).sum()
sqdist = lambda a, b: sqmag(a - b)

def dtanh(x):
    e = x.clip(-12, 12)
    np.exp(-2.*e, e)
    return 4*e / np.square(e + 1)

def ddtanh(x):
    return -2.*np.tanh(x)*dtanh(x)

def ldot(m1, m2):
    return np.sum(m1 * m2)
    
m1ldot = np.dot
m2ldot = m3ldot = ldot

def m2dotrows(m1, m2):
    assert (m1.ndim == m2.ndim == 2)
    return (m1 * m2).sum(1)

def normrows(m):
    for r in m:
        r /= sqrt(sqmag(r))

def copy_normrows(m): # for testing
    x=m.copy()
    normrows(x)
    return x

def m2dotm1(m1, m2, res = None, accumulate=False):
    assert (m1.ndim == 2 and m2.ndim == 1)
    if res is None:  res    = np.dot(m1, m2)
    elif accumulate: res   += np.dot(m1, m2)
    else:            res[:] = np.dot(m1, m2)
    return res

def m2kdotmk(m1, m2, res=None, accumulate=False):
    k = m2.ndim
    assert (m1.ndim == 2*k)
    if res is None: res = np.zeros(m1.shape[:k], m1.dtype)
    assert (res.shape == m1.shape[:k] and m2.shape == m1.shape[k:])
    if accumulate:
        for i in np.ndindex(res.shape):
            res[i] += ldot(m1[i], m2)
    else:
        for i in np.ndindex(res.shape):
            res[i]  = ldot(m1[i], m2)
    return res

m4dotm2 = m6dotm3 = m2kdotmk


def mkextmk(m1, m2, res=None, accumulate=False):
    k = m1.ndim
    assert (k == m2.ndim)
    if res is None: res = np.zeros(m1.shape + m2.shape, m1.dtype)
    assert (res.shape[:k] == m1.shape and res.shape[k:] == m2.shape)
    if accumulate:
        for i in np.ndindex(res.shape):
            res[i] += m1[i[:k]] * m2[i[k:]]
    else:
        for i in np.ndindex(res.shape):
            res[i]  = m1[i[:k]] * m2[i[k:]]
    return res
m2extm2 = m3extm3 = mkextmk

##### no gofast versions for these yet --

def thresh_less(m1, m2, thresh, out=None, accumulate=False):
    ''' out_i = m1_i if m2_i >= thresh, else 0 '''
    res = m1 * (m2 >= thresh)
    if out is None: return res
    if accumulate: out   += res
    else:          out[:] = res
    return out


try:
    from gofast.vecmath  import *
    fast = True
except ImportError:
    fast = False


