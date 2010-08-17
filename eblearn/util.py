import sys
import numpy as np
import pickle

from eblearn import lush_mat as lm

np.seterr('raise')
np.seterr(under = 'ignore')

try:
    from eblearn.gofast.util import rtype
except ImportError:
    print "+++ Warning: gofast isn't compiled."
    rtype = float

class ref(object):
    __slots__ = ('contents')
    def __init__(self, v): self.contents = v
    def __getstate__(self): return self.contents
    def __setstate__(self, v): self.contents=v

array = lambda items, dtype=rtype: np.array(items, dtype)
empty = lambda shape, dtype=rtype: np.empty(shape, dtype, 'C')
zeros = lambda shape, dtype=rtype: np.zeros(shape, dtype, 'C')
ones  = lambda shape, dtype=rtype: np.ones(shape,  dtype, 'C')

if rtype == float:
    random = np.random.random
else:
    random = lambda shape: np.random.random(shape).astype(rtype)


def ensure_dims(x, d):
    need = d - len(x.shape)
    if need > 0: return x.reshape(x.shape + ((1,) * need))
    return x

def ensure_tuple(x):
    if isinstance(x, tuple): return x
    if isinstance(x, int):   return (x,)
    return tuple(x)

def save_object(obj, dest):
    if type(dest) == str: dest = open(dest, 'wb')
    pickle.dump(obj, dest, protocol = pickle.HIGHEST_PROTOCOL)

def load_object(loc):
    if type(loc) == str: loc = open(loc, 'rb')
    return pickle.load(loc)

class agenerator (object):
    def __init__(self, gen, dtype=rtype, count=-1):
        self.gen = gen
        self.dtype = dtype
        self.count = count
    def __array__(self):
        return np.fromiter(self.gen(), self.dtype, self.count)
    def __iter__(self):
        return self.gen()

class abuffer (object):
    def __init__(self, n=(100,), dtype=rtype):
        n = ensure_tuple(n)
        assert (len(n) > 0 and np.prod(n) > 0), 'initial size must be positive'
        self.buf = np.empty(n, dtype=dtype)
        self.len = 0
    def __array__(self):
        return self.buf[:self.len]
    def _resize_buf(self, newsize):
        try:
            self.buf.resize(newsize)
        except ValueError:
            newbuf = np.empty(newsize, dtype=self.buf.dtype)
            newbuf[:self.len] = self.buf
            self.buf = newbuf
    def append(self, x):
        if len(self.buf) == self.len:
            self._resize_buf((self.len * 2,)+self.buf.shape[1:])
        self.buf[self.len] = x
        self.len += 1
    def extend(self, other):
        other = np.asarray(other)
        assert (self.buf.shape[1:] == other.shape[1:])
        olen = len(other)
        if olen > len(self.buf) - self.len:
            newlen = len(self.buf) + max(len(self.buf), olen)
            self._resize_buf((newlen,)+self.buf.shape[1:])
        self.buf[self.len:self.len+olen] = other
        self.len += olen
    def __len__    (self):       return self.len
    def __getitem__(self, i):    return self.__array__().__getitem__(i)
    def __setitem__(self, i, v): return self.__array__().__setitem__(i, v)
    def __str__    (self):       return str(self.__array__())
    def __repr__   (self):       return repr(self.__array__())

class abuffer_disk (object):
    def __init__(self, f, shape=(), dtype=rtype):
        if type(f) == str: f = open(f, 'w+b')
        shape = ensure_tuple(shape)
        self.len   = 0
        self.shape = shape
        self.dtype = dtype
        self.f     = f
        self.arr   = None
        self.update_header()
    
    def update_header(self):
        self.f.seek(0, 0) # seek start
        lm.save_matrix_header((self.len,)+self.shape,
                              np.dtype(self.dtype), self.f)
        self.f.seek(0, 2) # seek end
    
    def __array__(self):
        if self.arr is not None: return self.arr
        self.update_header()
        self.f.seek(0, 0)
        self.arr = lm.map_matrix(self.f, mode='r+')
        self.f.seek(0, 2)
        return self.arr
    
    def append(self, x):
        self.extend([x])
    
    def extend(self, xs):
        if len(xs) == 0: return
        self.arr = None
        xs = np.ascontiguousarray(xs, dtype=self.dtype)
        assert (xs.shape[1:] == self.shape), "shapes don't match"
        self.f.write(xs.data)
        self.len += len(xs)
    
    def __del__(self):
        self.update_header()
    
    def __len__    (self):       return self.len
    def __getitem__(self, i):    return self.__array__().__getitem__(i)
    def __setitem__(self, i, v): return self.__array__().__setitem__(i, v)
    def __str__    (self):       return str(self.__array__())
    def __repr__   (self):       return repr(self.__array__())
            

class rolling_average (object):
    def __new__(cls, shape, dtype=rtype):
        shape = ensure_tuple(shape)
        if shape[0] == 1:
            return degen_rolling_average(shape, dtype)
        
        return object.__new__(cls)
    
    def __init__(self, shape, dtype=rtype):
        self.buf = np.zeros(shape, dtype)
        self.clear()
    
    def _zero(self):
        buf = self.buf
        if buf.ndim == 1: return 0
        return np.zeros(buf.shape[1:], buf.dtype)
    
    def clear(self):
        self._avg = self._zero()
        self.pos  = 0
        self.full = False
    
    def append(self, val):
        buf, pos = self.buf, self.pos
        n = float(len(buf))
        
        if self.full: self._avg += (val - buf[pos]) / n
        else:         self._avg += val / n
        buf[pos] = val
        
        self.pos = (pos + 1) % len(buf)
        if self.pos == 0: self.full = True
    
    def average(self, redo=False):
        if self.full:
            if redo: self._avg = self.buf.mean(0)
            return self._avg
        
        buf, pos = self.buf, self.pos
        n   = float(len(buf))
        if pos == 0:
            return self._zero()
        
        if redo: self._avg = buf[:pos].sum(0) / n
        return self._avg * (n / pos)

class degen_rolling_average (rolling_average):
    def __new__(cls, shape, dtype=rtype):
        return object.__new__(cls)
    def __init__(self, shape, dtype=rtype):
        shape = ensure_tuple(shape)
        assert (shape[0] == 1)
        super(degen_rolling_average, self).__init__(shape, dtype)
    def append(self, val):
        self._avg = val
        self.full = True
    def average(self, redo=False):
        return self._avg


ENABLE_BREAKPOINTS = True
def enable_breaks(b = True):
    global ENABLE_BREAKPOINTS
    ENABLE_BREAKPOINTS = b

def debug_break(msg = None):
    if not ENABLE_BREAKPOINTS: return
    if msg: msg = 'DEBUG BREAK: %s' % (msg,)
    else:   msg = 'DEBUG BREAK'
    print '==================================='
    print msg
    print '==================================='
    try:
        raise ImportError() # buggy outside if interactive IPython sessions
        import IPython.Debugger; 
        t = IPython.Debugger.Tracer()
        t.debugger.set_trace(sys._getframe().f_back)
    except ImportError:
        import pdb; 
        p = pdb.Pdb()
        p.set_trace(sys._getframe().f_back)

def replace_global(oldver, newver, packages):
    assert (oldver not in (None, (), [], False, True))
    assert (not isinstance(oldver, (int,long,float)))
    for mod in sys.modules.itervalues():
        mod_package = getattr(mod, '__package__', None)
        if mod_package is None: continue
        if any((mod_package == pkg or mod_package.startswith(pkg+'.'))
               for pkg in packages):
            changes = {}
            for k, v in mod.__dict__.iteritems():
                if v is oldver: changes[k] = newver
            mod.__dict__.update(changes)

