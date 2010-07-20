import sys
import scipy as sp
import pickle
from math import pi, sqrt
from _util import rtype

sp.seterr('raise')

array = lambda items: sp.array(items, rtype)
empty = lambda shape: sp.empty(shape, rtype, 'C')
zeros = lambda shape: sp.zeros(shape, rtype, 'C')
ones  = lambda shape: sp.ones(shape, rtype, 'C')
product = sp.prod
sqmag  = lambda x: sp.square(x).sum()
sqdist = lambda a, b: sqmag(a - b)

def dtanh(x):
    e = x.clip(-8, 8)
    sp.exp(-2.*e, e)
    return 4*e / sp.square(e + 1)

def thresh_less(inp, thresh, out=None):
    if out is None: out    = inp.copy()
    else:           out[:] = inp
    out *= (inp >= thresh)
    return out

def ensure_dims(x, d):
    need = d - len(x.shape)
    if need > 0: return x.reshape(x.shape + ((1,) * need))
    return x

def ensure_tuple(x):
    try:    return tuple(x)
    except: return (x,)

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



class around_super(object):
    ''' Avoid duplicate "around" calls when using super() '''
    def __new__(self, cls, *args):
        if not hasattr(cls, '__renames__') or not args:
            return _base_super(cls, *args)
        return object.__new__(self)
    def __init__(self, cls, obj):
        self.__sup     = _base_super(cls, obj)
        self.__renames = cls.__renames__
    def __getattribute__(self, name):
        if name.startswith('_around_super__'):
            return object.__getattribute__(self, name)
        sup     = self.__sup
        renames = self.__renames
        if name in renames: name = renames[name]
        return getattr(sup, name)

try:
    if not hasattr(__builtins__, '_base_super'):
        __builtins__._base_super    = super
    __builtins__.super       = around_super
except:
    # pdb does this
    if '_base_super' not in __builtins__:
        __builtins__['_base_super'] = super
    __builtins__['super']       = around_super



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


import Tkinter as tk
import Image   as image
import tempfile 
import os
    



tk_roots    = {}
tk_def_root = None
tk_cur_root = None
tk_max      = 1
def set_window(id = None):
    global tk_roots, tk_def_root, tk_cur_root
    if id is None: id = tk_def_root
    if id is None: new_window()
    else:
        assert (id in tk_roots)
        tk_cur_root = id

def new_window(width=800, height=600, title = 'Python Window'):
    global tk_roots, tk_cur_root, tk_def_root, tk_max
    
    import Tkinter as tk
    root = tk.Tk(); root.geometry('%dx%d'%(width, height)); root.title(title)

    id = tk_max; tk_max +=1
    root.id = id

    tk_roots[id] = root
    if tk_def_root is None: 
        tk_def_root = id
    tk_cur_root = id
    
    canvas = tk.Canvas(root, width=width, height=height, bg = 'white')
    canvas.pack(expand=tk.YES, fill=tk.BOTH)
    canvas.images = []
    
    root.canvas = canvas
    
    def root_destr():
        global tk_roots, tk_def_root, tk_cur_root
        id = root.id
        cls(id); canvas.destroy(); root.destroy(); del tk_roots[id]
        next_id = None if not tk_roots else tk_roots.keys()[0]
        if tk_def_root == id: tk_def_root = next_id
        if tk_cur_root == id: tk_cur_root = tk_def_root
    
    root.protocol("WM_DELETE_WINDOW", root_destr)
    return root.id

def cls(id = None):
    ''' clear window '''
    global tk_roots, tk_cur_root
    if id is None: id = tk_cur_root
    if id is None: return
    canvas = tk_roots[id].canvas
    for i in canvas.find_all(): canvas.delete(i)
    canvas.images = []

def draw_mat(mat, x=0, y=0, minv=None, maxv=None, scale=1.0):
    global tk_roots, tk_cur_root
    assert (len(mat.shape) == 2 or (len(mat.shape) == 3 and mat.shape[2] == 3))
    if tk_cur_root is None: new_window()

    if minv is None: minv = mat.min()
    if maxv is None: maxv = mat.max()

    if maxv - minv < 1e-9:
        print '+++ Warning (draw_mat): maxv = minv'
        mat = sp.zeros(mat.shape, sp.uint8)
    else:
        mat = (mat.clip(minv, maxv) - minv) * (255. / (maxv - minv))
        mat = mat.astype(sp.uint8)
    
    img = image.fromarray(mat)
    if scale != 1.0:
        width, height = img.size
        img = img.resize((width * scale, height * scale))

    # Python mangles the image if we try to send it directly
    fname = tempfile.mktemp('.ppm', 'tkimg_')
    img.save(fname, 'PPM')
    
    root = tk_roots[tk_cur_root]
    tkimg = tk.PhotoImage(file=fname, format='PPM', master=root)
    os.unlink(fname)

    root.canvas.create_image(x, y, anchor=tk.NW, image=tkimg)
    root.canvas.images.append(tkimg)

def draw_lines(x1, y1, x2, y2, *rest):
    global tk_roots, tk_cur_root
    if tk_cur_root is None: new_window()

    root = tk_roots[tk_cur_root]
    root.canvas.create_line(x1, y1, x2, y2, *rest)

def draw_text(x, y, text):
    global tk_roots, tk_cur_root
    if tk_cur_root is None: new_window()

    root = tk_roots[tk_cur_root]
    root.canvas.create_text(x, y, text=text, anchor=tk.SW)
    
