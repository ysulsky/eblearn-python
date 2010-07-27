import sys
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
    try:    return tuple(x)
    except: return (x,)


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


    
