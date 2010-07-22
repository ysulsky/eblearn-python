import sys
import scipy as sp
import os, pickle, re, tempfile
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

from inspect import CO_VARARGS, CO_VARKEYWORDS, getargspec

class around_methods (type):
    ''' metaclass that supports around methods (@around decorator) '''

    @staticmethod
    def nicify(inner, outer):
        outer.__doc__  = inner.__doc__ or ''
        outer.__name__ = inner.__name__
        
        outer.__dict__.update(inner.__dict__)
        ofcode = outer.func_code
        ifcode = inner.func_code

        # This doesn't work yet -- trying to get help() to show the
        # correct arguments. Will shove it in __doc__ instead
        #
        # outer_nargs = ofcode.co_argcount
        # if ofcode.co_flags & CO_VARARGS:     outer_nargs += 1
        # if ofcode.co_flags & CO_VARKEYWORDS: outer_nargs += 1

        # inner_nargs = ofcode.co_argcount
        # if ofcode.co_flags & CO_VARARGS:     inner_nargs += 1
        # if ofcode.co_flags & CO_VARKEYWORDS: inner_nargs += 1

        # optflags  = CO_VARARGS | CO_VARKEYWORDS
        # rflags    = (ofcode.co_flags &~optflags)|(ifcode.co_flags &optflags)
        # rnlocals  = ofcode.co_nlocals - outer_nargs + inner_nargs
        # rvarnames = (ifcode.co_varnames[:inner_nargs]
        #              + ofcode.co_varnames[outer_nargs:])
        # rargcount = ifcode.co_argcount

        argspec = getargspec(inner)
        defaults = argspec.defaults or ()
        n_nodefault = len(argspec.args) - len(defaults)
        args = argspec.args[:n_nodefault] + ['%s=%s' % (arg, repr(val))
                                             for (arg, val) 
                                             in zip(argspec.args[n_nodefault:],
                                                    defaults)]
        if argspec.varargs:  args.append('*'+argspec.varargs)
        if argspec.keywords: args.append('**'+argspec.keywords)
        
        usage = 'Usage: %s(%s)' % (outer.__name__, ', '.join(args))

        # don't screw up indentation in the __doc__
        doclines = outer.__doc__.splitlines()
        isize = lambda s: len(s[:re.match('^\s*',s).span()[1]].expandtabs())
        while doclines and not doclines[0].strip(): doclines = doclines[1:]
        if len(doclines) > 0:
            usage = ' '*isize(doclines[0]) + usage
        if len(doclines) > 1:
            indent = min([0]+[isize(l) for l in doclines[1:] if l.strip()])
            doclines[0] = ' '*indent + doclines[0].lstrip()
        outer.__doc__ = '%s\n\n%s' % (usage, '\n'.join(doclines))
        
        rflags    = ofcode.co_flags
        rnlocals  = ofcode.co_nlocals
        rvarnames = ofcode.co_varnames
        rargcount = ofcode.co_argcount
        outer.func_code = \
            ofcode.__class__ (rargcount,             rnlocals,
                              ofcode.co_stacksize,   rflags,
                              ofcode.co_code,        ofcode.co_consts,
                              ofcode.co_names,       rvarnames,
                              ifcode.co_filename,    ifcode.co_name,
                              ifcode.co_firstlineno, ofcode.co_lnotab,
                              ofcode.co_freevars,    ofcode.co_cellvars)

    @staticmethod
    def make_wrapper(inner_fn, around_fn):
        ret = lambda self, *args, **kwargs: \
            around_fn(self, inner_fn, *args, **kwargs)
        around_methods.nicify(inner_fn, ret)
        return ret

    def __new__ (cls, name, bases, dct):
        renames = {}
        dct['__renames__'] = renames
        for base in bases:
            if hasattr(base, '__renames__'):
                renames.update(base.__renames__)

        changes = {}
        overrides = []
        dummy = lambda *args, **kwargs: None
        for k, v in dct.iteritems():
            if isinstance(v, around):
                renames[k] = '_inner__' + k
                changes['_inner__'+k]  = v.inner_fn
                changes['_around__'+k] = v.around_fn
                changes[k] = around_methods.make_wrapper(v.inner_fn,v.around_fn)
            elif k in renames:
                overrides.append(k)
                changes[renames[k]] = v
                changes[k]          = dummy
        for k in overrides: del dct[k]
        dct.update(changes)
        dct['__renamed_overrides__'] = overrides
        return type.__new__(cls, name, bases, dct)

    def __init__(cls, name, bases, dct):
        for k in cls.__renamed_overrides__:
            inner_fn  = getattr(cls, '_inner__'+k)
            around_fn = getattr(cls, '_around__'+k)
            setattr(cls, k, around_methods.make_wrapper(inner_fn, around_fn))
        return type.__init__(cls, name, bases, dct)


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

def ensure_window(width=None, height=None, title=None):
    global tk_roots, tk_cur_root
    r = tk_cur_root
    if r is None:
        kwargs = {}
        if width:  kwargs['width']=width
        if height: kwargs['height']=height
        if title:  kwargs['title']=title
        new_window(**kwargs)
    else:
        r = tk_roots[r]
        if width or height:
            if not height: height = r.geometry().split('+')[0].split('x')[1]
            if not width:   width = r.geometry().split('+')[0].split('x')[0]
            r.geometry('%sx%s' % (width, height))
        if title:
            r.title(title)

def new_window(width=800, height=600, title = 'Python Window'):
    global tk_roots, tk_cur_root, tk_def_root, tk_max
    
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
    root.canvas.create_text(x, y, text=text, anchor=tk.SW, font=('courier',8))
    
