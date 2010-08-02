import re
from inspect import CO_VARARGS, CO_VARKEYWORDS, getargspec

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
        type.__init__(cls, name, bases, dct)


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

