from eblearn import *
import scipy as sp

class eb_dsource (object):
    def __init__(self): self.current = 0
    def size(self):     return 0
    def seek(self, n):  self.current = n % self.size()
    def next(self):     self.current = (self.current + 1) % self.size()
    def tell(self):     return self.current
    def fprop(self, input, output): raise NotImplementedError()
    def shape(self):
        inp = state(()); outp = state(())
        self.fprop(inp, outp)
        return (inp.shape, outp.shape)

class dsource_unsup (eb_dsource):
    def __init__(self, inputs, bias = 0., coeff = 1.):
        self.current = 0
        self.inputs  = ensure_dims(inputs, 2)
        self.bias    = bias
        self.coeff   = coeff
        self.shuffle = sp.array(xrange(len(inputs)))
        sp.random.shuffle(self.shuffle)

    def size(self): return len(self.inputs)

    def normalize(self, scalar_bias = False, scalar_coeff = False):
        avg = self.inputs.mean(None if scalar_bias  else 0)
        self.bias  = -avg
        std = self.inputs.std (None if scalar_coeff else 0).clip(1e-6)
        self.coeff = 1.0 / std

    def _fprop_input(self, input):
        x = (self.inputs[self.shuffle[self.current]] + self.bias) * self.coeff
        input.resize(x.shape)
        input.x[:] = x
        
    def fprop(self, input, output):
        self._fprop_input(input)
        
        output.resize(input.shape)
        output.x[:] = input.x

class dsource_sup (dsource_unsup):
    def __init__(self, inputs, targets, bias = 0., coeff = 1.):
        super(dsource_sup, self).__init__(inputs, bias, coeff)
        self.targets = ensure_dims(targets, 2)

    def fprop(self, input, output):
        self._fprop_input(input)
        
        y = self.targets[self.shuffle[self.current]]
        output.resize(y.shape)
        output.x[:] = y
