import sys
import numpy as np
import scipy as sp
from math import pi, sqrt
from vecmath import *
from idx import *
from lush_mat import *

sp.seterr('raise')

try:
    from ui import *
except ImportError:
    print "+++ Warning: UI libraries not found."

try:
    from gofast._util import rtype
except ImportError:
    print "+++ Warning: gofast isn't compiled."
    rtype = sp.float64

class ref(object):
    __slots__ = ('contents')
    def __init__(self, v): self.contents = v
    def __getstate__(self): return self.contents
    def __setstate__(self, v): self.contents=v

array = lambda items: sp.array(items, rtype)
empty = lambda shape: sp.empty(shape, rtype, 'C')
zeros = lambda shape: sp.zeros(shape, rtype, 'C')
ones  = lambda shape: sp.ones(shape, rtype, 'C')

imshow = sp.misc.imshow
product = sp.prod

def ensure_dims(x, d):
    need = d - len(x.shape)
    if need > 0: return x.reshape(x.shape + ((1,) * need))
    return x

def ensure_tuple(x):
    if isinstance(x, tuple): return x
    if isinstance(x, int):   return (x,)
    return tuple(x)

class abuffer (object):
    def __init__(self, n=(100,), dtype=rtype, initial=None):
        n = ensure_tuple(n)
        assert (len(n) > 0 and np.prod(n) > 0), 'initial size must be positive'
        self._buf = np.empty(n, dtype=dtype)
        self._len = 0
        if initial is not None: self.extend(initial)
    def __array__(self):
        return self._buf[:self._len]
    def _resize_buf(self, newsize):
        try:
            self._buf.resize(newsize)
        except ValueError:
            newbuf = np.empty(newsize, dtype=self._buf.dtype)
            newbuf[:self._len] = self._buf
            self._buf = newbuf
    def append(self, x):
        if len(self._buf) == self._len:
            self._resize_buf((self._len * 2,)+self._buf.shape[1:])
        self._buf[self._len] = x
        self._len += 1
    def extend(self, other):
        other = np.asarray(other)
        assert (self._buf.shape[1:] == other.shape[1:])
        olen = len(other)
        if olen > len(self._buf) - self._len:
            newlen = len(self._buf) + max(len(self._buf), olen)
            self._resize_buf((newlen,)+self._buf.shape[1:])
        self._buf[self._len:self._len+olen] = other
        self._len += olen
    def __len__(self): return self._len
    def __getitem__(self, i):    return self.__array__().__getitem__(i)
    def __setitem__(self, i, v): return self.__array__().__setitem__(i, v)
    def __str__(self):           return str(self.__array__())
    def __repr__(self):          return repr(self.__array__())
        # return 'abuffer(%s, %s, %s)' % (self._bufsize,
        #                                 self._buf.dtype, self.__array__())


ENABLE_BREAKPOINTS = True
def enable_breaks(b = True):
    global ENABLE_BREAKPOINTS
    ENABLE_BREAKPOINTS = b

def debug_break(msg = None):
    if not ENABLE_BREAKPOINTS: return
    print '==================================='
    print 'DEBUG BREAK:', msg
    print '==================================='
    try:
        import IPython.Debugger; 
        t = IPython.Debugger.Tracer()
        t.debugger.set_trace(sys._getframe().f_back)
    except ImportError:
        import pdb; 
        p = pdb.Pdb()
        p.set_trace(sys._getframe().f_back)


    
