import sys
import numpy as np
import pickle

np.seterr('raise')

try:
    from gofast.util import rtype
except ImportError:
    print "+++ Warning: gofast isn't compiled."
    rtype = float

class ref(object):
    __slots__ = ('contents')
    def __init__(self, v): self.contents = v
    def __getstate__(self): return self.contents
    def __setstate__(self, v): self.contents=v

array = lambda items: np.array(items, rtype)
empty = lambda shape: np.empty(shape, rtype, 'C')
zeros = lambda shape: np.zeros(shape, rtype, 'C')
ones  = lambda shape: np.ones(shape, rtype, 'C')
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

def replace_global(package, oldver, newver):
    assert (oldver not in (None, (), [], False, True))
    assert (not isinstance(oldver, (int,long,float)))
    for mod in sys.modules.itervalues():
        mod_package = getattr(mod, '__package__', None)
        if mod_package is None: continue
        if mod_package == package or mod_package.startswith(package+'.'):
            changes = {}
            for k, v in mod.__dict__.iteritems():
                if v is oldver: changes[k] = newver
            mod.__dict__.update(changes)

