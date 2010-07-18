from util import *
from state import *
from parameter import *
from module import *

def always_init (fn_start, fn_end):
    """ Depending on the inheritence chain and calls to super().__init__
        fn_start and fn_end may be called several times """
    class always_init_meta (type):
        def __init__ (cls, name, bases, dct):
            cls.cur__init__ = cls.__init__
            def real_init (self, *args):
                fn_start(self)
                cls.cur__init__(self, *args)
                fn_end(self)
            cls.__init__ = real_init
    return always_init_meta

def ebmod_init_start(self):
    if eb_module.init_lvl == 0:
        print 'dynamic extent started'
        eb_module.cur_parameter = parameter()
    self.parameter = eb_module.cur_parameter
    eb_module.init_lvl += 1
    print '   ebmod_init_start'

def ebmod_init_end(self):
    print '   ebmod_init_end'
    eb_module.init_lvl -= 1
    if eb_module.init_lvl == 0:
        print 'dynamic extent ended'
        eb_module.cur_parameter = None

class eb_module (object):
    __metaclass__ = always_init(ebmod_init_start, ebmod_init_end)
    cur_parameter = None
    init_lvl = 0


    def has_params(self):
        return self.parameter is not None and self.parameter.size() > 0

    def forget(self):
        if self.has_params(): raise NotImplementedError()

    def normalize(self):
        if self.has_params(): raise NotImplementedError()
        
    def param(self, shape):
        s = state(shape)
        self.parameter.append(s)
        return s
    
    def _merge_parameters(self, other):
        if self.parameter is None: 
            self.parameter = other.parameter
        else:
            self.parameter.merge(other.parameter)


class module_1_1 (eb_module):
    def fprop(self, input, output):
        raise NotImplementedError()

    def bprop_input(self, input, output):
        raise NotImplementedError()
    def bprop_param(self, input, output):
        if self.has_params(): raise NotImplementedError()

    def bbprop_input(self, input, output):
        raise NotImplementedError()
    def bbprop_param(self, input, output):
        if self.has_params(): raise NotImplementedError()

    def bprop(self, input, output):
        self.bprop_input(input, output)
        self.bprop_param(input, output)

    def bbprop(self, input, output):
        self.bbprop_input(input, output)
        self.bbprop_param(input, output)


class module_2_1 (eb_module):
    def fprop(self, input1, input2, output):
        raise NotImplementedError()

    def bprop_input(self, input1, input2, output):
        raise NotImplementedError()
    def bprop_param(self, input1, input2, output):
        if self.has_params(): raise NotImplementedError()

    def bbprop_input(self, input1, input2, output):
        raise NotImplementedError()
    def bbprop_param(self, input1, input2, output):
        if self.has_params(): raise NotImplementedError()

    def bprop(self, input1, input2, output):
        self.bprop_input(input1, input2, output)
        self.bprop_param(input1, input2, output)

    def bbprop(self, input, output):
        self.bbprop_input(input1, input2, output)
        self.bbprop_param(input1, input2, output)

