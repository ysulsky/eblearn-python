from util import *
from state import *
from parameter import *
from module import *

class eb_module (object):
    __metaclass__ = around_methods
    cur_parameter = None
    init_lvl = 0
    
    def _init_around(self, inner_init, *args, **kwargs):
        if eb_module.init_lvl == 0:
            #print 'dynamic extent started'
            eb_module.cur_parameter = parameter()
        self._parameter = eb_module.cur_parameter
        self._forget_param = None
        eb_module.init_lvl += 1
        #print '   ebmod_init_start'
    
        inner_init(self, *args, **kwargs)
    
        #print '   ebmod_init_end'
        eb_module.init_lvl -= 1
        if eb_module.init_lvl == 0:
            #print 'dynamic extent ended'
            eb_module.cur_parameter = None

    @around(_init_around)
    def __init__(self): pass

    def has_params(self):
        #XXX: not exact. use the "no_params" subclass to get this right
        return self._parameter.size() > 0

    def _forget_around(self, old_forget, *args, **kwargs):
        self._parameter.reset()
        old_forget(self, *args, **kwargs)

    @around(_forget_around)
    def forget(self):
        raise NotImplementedError()

    def normalize(self):
        raise NotImplementedError()
        
    def param(self, shape):
        s = state(shape)
        self._parameter.append(s)
        return s
    
    def _merge_parameters(self, other):
        self._parameter.merge(other._parameter)

    def _get_parameter(self):
        return self._parameter
    
    parameter = property(_get_parameter)

    def _get_forget_param(self):
        if self._forget_param is not None: return self._forget_param
        return self._parameter.forget
    def _set_forget_param(self, v):
        self._forget_param = v

    forget_param = property(_get_forget_param, _set_forget_param)
    

class no_params (object):
    def param(self, shape):  assert('No parameters allowed' == 0)
    def bprop_param(*args):  pass
    def bbprop_param(*args): pass
    def forget(self):        pass
    def normalize(self):     pass
    def has_params(self):    return False

class module_1_1 (eb_module):
    def fprop(self, input, output):
        raise NotImplementedError()

    def bprop_input(self, input, output):
        raise NotImplementedError()
    def bprop_param(self, input, output):
        raise NotImplementedError()

    def bbprop_input(self, input, output):
        raise NotImplementedError()
    def bbprop_param(self, input, output):
        raise NotImplementedError()

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
        raise NotImplementedError()

    def bbprop_input(self, input1, input2, output):
        raise NotImplementedError()
    def bbprop_param(self, input1, input2, output):
        raise NotImplementedError()

    def bprop(self, input1, input2, output):
        self.bprop_input(input1, input2, output)
        self.bprop_param(input1, input2, output)

    def bbprop(self, input1, input2, output):
        self.bbprop_input(input1, input2, output)
        self.bbprop_param(input1, input2, output)

