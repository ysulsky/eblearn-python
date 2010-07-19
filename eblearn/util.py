import sys
import scipy as sp
import pickle
from math import sqrt, log, tanh
from _util import rtype

sp.seterr('raise')

array = lambda items: sp.array(items, rtype)
empty = lambda shape: sp.empty(shape, rtype, 'C')
zeros = lambda shape: sp.zeros(shape, rtype, 'C')
ones  = lambda shape: sp.ones(shape, rtype, 'C')
product = sp.prod
sqdist = lambda a, b: ((a - b) ** 2).sum()

def dtanh(x):
    e = x.clip(-8, 8)
    sp.exp(-2.*e, e)
    return 4*e / (e + 1) ** 2

def thresh_less(inp, thresh, out=None):
    if out is None: out    = inp.copy()
    else:           out[:] = inp
    out *= (inp >= thresh)
    return out

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


base_super = super
class around_super(object):
    ''' Avoid duplicate "around" calls when using super() '''
    def __new__(self, cls, *args):
        if not hasattr(cls, '__renames__') or not args:
            return base_super(cls, *args)
        return object.__new__(self)
    def __init__(self, cls, obj):
        self.__sup     = base_super(cls, obj)
        self.__renames = cls.__renames__
    def __getattribute__(self, name):
        if name.startswith('_around_super__'):
            return object.__getattribute__(self, name)
        sup     = self.__sup
        renames = self.__renames
        if name in renames: name = renames[name]
        return getattr(sup, name)

try:
    __builtins__.super = around_super
except:
    # pdb does this
    __builtins__['super'] = around_super


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
