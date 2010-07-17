import scipy as sp

class eb_dsource (object):
    def __init__(self): self.current = 0
    def size(self):     return 0
    def seek(self, n):  self.current = n % self.size()
    def next(self):     self.current = (self.current + 1) % self.size()
    def tell(self):     return self.current
    def fprop(input, output): raise NotImplementedError()

class dsource_unsup (eb_dsource):
    def __init__(self, inputs, bias, coeff):
        self.current = 0
        self.inputs  = inputs
        self.bias    = bias
        self.coeff   = coeff
        self.shuffle = sp.array(xrange(len(inputs)))
        sp.random.shuffle(self.shuffle)

    def size(self): return len(inputs)
    def fprop(input, output):
        x = self.inputs[self.shuffle[self.current], :]
        x = (x + bias) * coeff
        
        input.resize(x.shape)
        input[:] = x
        
        output.resize(x.shape)
        output[:] = x
