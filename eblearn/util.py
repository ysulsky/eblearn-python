import sys
import scipy as sp
import pickle
from math import sqrt, log
from _util import rtype

array = lambda items: sp.array(items, rtype)
empty = lambda shape: sp.empty(shape, rtype, 'C')
zeros = lambda shape: sp.zeros(shape, rtype, 'C')
ones  = lambda shape: sp.ones(shape, rtype, 'C')
product = sp.prod
sqdist = lambda a, b: ((a - b) ** 2).sum()

def ensure_dims(x, d):
    need = d - len(x.shape)
    if need > 0: return x.reshape(x.shape + ((1,) * need))
    return x


class around (object):
    ''' decorator for making "around" functions '''
    def __init__(self, around_fn):
        self.around_fn = around_fn
        self.inner_fn  = None
    def __call__(self, inner_fn):
        self.inner_fn  = inner_fn
        return self

class around_methods (type):
    ''' metaclass that supports around methods (@around decorator) '''
    def __new__ (cls, name, bases, dct):
        def make_wrapper(inner_name, around_fn):
            return lambda self, *args, **kwargs: \
                around_fn(self, getattr(self, inner_name), *args, **kwargs)
        renames = {}
        dct['__renames__'] = renames
        for base in bases:
            if hasattr(base, '__renames__'):
                renames.update(base.__renames__)

        changes = {}
        todel   = []
        for k, v in dct.iteritems():
            if isinstance(v, around):
                renames[k] = '_inner__' + k
                changes['_inner__'+k] = v.inner_fn
                changes[k] = make_wrapper('_inner__'+k, v.around_fn)
            elif k in renames:
                todel.append(k)
                changes[renames[k]] = v
        for k in todel: del dct[k]
        dct.update(changes)
        return type.__new__(cls, name, bases, dct)


ENABLE_BREAKPOINTS = True
def enable_breaks(b = True):
    global ENABLE_BREAKPOINTS
    ENABLE_BREAKPOINTS = b

def debug_break():
    if not ENABLE_BREAKPOINTS: return
    try:
        import IPython.Debugger; 
        t = IPython.Debugger.Tracer()
        t.debugger.set_trace(sys._getframe().f_back)
    except ImportError:
        import pdb; 
        p = pdb.Pdb()
        p.set_trace(sys._getframe().f_back)
