from module import *

class linear (module_1_1):
    def __init__(self, shape_in, shape_out):
        self.shape_in  = ensure_tuple(shape_in)
        self.shape_out = ensure_tuple(shape_out)
        size_in  = product(shape_in)
        size_out = product(shape_out)
        self.w = self.param((size_out, size_in))

    def forget(self):
        arg = self.parameter.forget
        fanin = self.w.shape[1]
        z = arg.lin_value / (fanin ** (1.0 / arg.lin_exponent))
        self.w.x = sp.random.random(self.w.shape) * (2*z) - z

    def normalize(self):
        for col in self.w.x.T:
            col *= 1.0 / sqrt(sqmag(col))

    def fprop(self, input, output):
        assert (self.shape_in  == input.shape)
        output.resize(self.shape_out)
        output.x[:] = sp.dot(self.w.x, input.x.ravel()).reshape(output.shape)

    def bprop_input(self, input, output):
        input.dx.ravel()[:] += sp.dot(self.w.x.T, output.dx.ravel())
    def bprop_param(self, input, output):
        self.w.dx += sp.outer(output.dx.ravel(), input.x.ravel())

    def bbprop_input(self, input, output):
        input.ddx.ravel()[:] += sp.dot(sp.square(self.w.x.T),output.ddx.ravel())
    def bbprop_param(self, input, output):
        self.w.ddx += sp.outer(output.ddx.ravel(), sp.square(input.x.ravel()))


class bias (module_1_1):
    def __init__(self, shape_in):
        self.b = self.param((shape_in))

    def forget(self):
        arg = self.parameter.forget
        z = arg.lin_value
        self.b.x = sp.random.random(self.b.shape) * (2*z) - z

    def normalize(self):
        self.b.x *= 1.0 / sqrt(sqmag(self.b.x))
    
    def fprop(self, input, output):
        assert (self.b.shape  == input.shape)
        output.resize(self.b.shape)
        output.x[:] = input.x + self.b.x

    def bprop_input(self, input, output):
        input.dx  += output.dx
    def bprop_param(self, input, output):
        self.b.dx += output.dx

    def bbprop_input(self, input, output):
        input.ddx  += output.ddx
    def bbprop_param(self, input, output):
        self.b.ddx += output.ddx


