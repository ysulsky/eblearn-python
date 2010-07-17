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
    self.has_params = False
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

    def forget(self):
        if self.has_params:
            raise NotImplementedError()
        
    def param(self, shape):
        s = state(shape)
        self.parameter.append(s)
        self.has_params = True
        return s

class module_1_1 (eb_module):
    def fprop(self, input, output):
        raise NotImplementedError()

    def bprop_param(self, input, output):
        raise NotImplementedError()
    def bprop_input(self, input, output):
        raise NotImplementedError()

    def bbprop_param(self, input, output):
        raise NotImplementedError()
    def bbprop_input(self, input, output):
        raise NotImplementedError()

    def bprop(self, input, output):
        self.bprop_param(input, output)
        self.bprop_input(input, output)

    def bbprop(self, input, output):
        self.bbprop_param(input, output)
        self.bbprop_input(input, output)


class linear (module_1_1):
    def __init__(self, shape_in, shape_out):
        self.shape_in  = shape_in
        self.shape_out = shape_out
        size_in  = product(shape_in)
        size_out = product(shape_out)
        self.w = self.param((size_out, size_in))

    def forget(self):
        arg = self.parameter.forget
        fanin = self.w.shape[1]
        z = arg.lin_value / (fanin ** (1.0 / arg.lin_exponent))
        self.w.x = sp.random.random(self.w.shape) * (2*z) - z

    def fprop(self, input, output):
        assert (self.shape_in  == input.shape)
        output.resize(self.shape_out)
        output.x = sp.dot(self.w.x, input.x.flat).reshape(output.shape)

    def bprop_param(self, input, output):
        self.w.dx += sp.outer(output.dx.flat, input.x.flat)

    def bprop_input(self, input, output):
        input.dx.flat += sp.dot(self.w.x.T, output.dx.flat)

    def bbprop_param(self, input, output):
        self.w.ddx += sp.outer(output.ddx.flat ** 2, input.x.flat)
    
    def bbprop_input(self, input, output):
        input.ddx.flat += sp.dot(self.w.x.T ** 2, output.ddx.flat)


class bias (module_1_1):
    def __init__(self, shape_in):
        self.b = self.param((shape_in))

    def forget(self):
        arg = self.parameter.forget
        z = arg.lin_value
        self.b.x = sp.random.random(self.b.shape) * (2*z) - z
    
    def fprop(self, input, output):
        assert (self.b.shape  == input.shape)
        output.resize(self.b.shape)
        output.x = input.x + self.b.x

    def bprop_param(self, input, output):
        self.b.dx += output.dx

    def bprop_input(self, input, output):
        input.dx  += output.dx

    def bbprop_param(self, input, output):
        self.b.ddx += output.ddx

    def bbprop_input(self, input, output):
        input.ddx  += output.ddx

