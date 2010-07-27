from module import *
from arch import *

class transfer_identity (no_params, module_1_1):
    def fprop(self, input, output):
        output.resize(input.shape)
        output.x[:] = input.x
    def bprop_input (self, input, output):
        input.dx  += output.dx
    def bbprop_input(self, input, output):
        input.ddx += output.ddx

class transfer_square (no_params, module_1_1):
    def fprop(self, input, output):
        output.resize(input.shape)
        sp.square(input.x, output.x)
    def bprop_input (self, input, output):
        input.dx  += output.dx * 2 * input.x
    def bbprop_input(self, input, output):
        input.ddx += output.ddx * 4 * sp.square(input.x)
        input.ddx += output.dx * 2

class transfer_cube (no_params, module_1_1):
    def fprop(self, input, output):
        output.resize(input.shape)
        output.x[:] = input.x ** 3
    def bprop_input(self, input, output):
        input.dx += output.dx * 3 * sp.square(input.x)
    def bbprop_input(self, input, output):
        input.ddx += output.ddx * 9 * (input.x **4)
        input.ddx += output.dx * 6 * input.x

class transfer_exp (no_params, module_1_1):
    def fprop(self, input, output):
        output.resize(input.shape)
        sp.exp(input.x, output.x)
    def bprop_input(self, input, output):
        input.dx += output.dx * sp.exp(input.x)
    def bbprop_input(self, input, output):
        expin = sp.exp(input.x)
        input.ddx += output.ddx * (expin ** 2)
        input.ddx += output.dx * expin

class transfer_tanh (no_params, module_1_1):
    def fprop(self, input, output):
        output.resize(input.shape)
        sp.tanh(input.x, output.x)
    def bprop_input (self, input, output):
        input.dx  += output.dx  * dtanh(input.x)
    def bbprop_input(self, input, output):
        input.ddx += output.ddx * dtanh(input.x) ** 2
        input.ddx += output.dx * ddtanh(input.x)

class transfer_abs (no_params, module_1_1):
    def __init__(self, thresh = 0.0001):
        self.thresh = thresh
    def fprop(self, input, output):
        output.resize(input.shape)
        sp.absolute(input.x, output.x)
    def bprop_input (self, input, output):
        sx = None
        if self.thresh != 0:
            sx = input.x / self.thresh
            sp.trunc(sx, sx)
            sp.sign(sx, sx)
        else:
            sx = sp.sign(input.x)
        input.dx += sx * output.dx
    def bbprop_input(self, input, output):
        input.ddx += output.ddx

class transfer_copy_flipsign (no_params, module_1_1):
    def fprop(self, input, output):
        in_len    = len(input)
        out_shape = (2 * in_len,) + input.shape[1:]
        output.resize(out_shape)
        output.x[:in_len] = input.x
        output.x[in_len:] = -input.x
    def bprop_input (self, input, output):
        in_len     = len(input)
        input.dx  += output.dx[:in_len]
        input.dx  -= output.dx[in_len:]
    def bbprop_input(self, input, output):
        in_len     = len(input)
        input.ddx += output.ddx[:in_len]
        input.ddx += output.ddx[in_len:]

class transfer_greater (no_params, module_1_1):
    def __init__(self, fwd_thresh = 0., bwd_thresh = 0.0001):
        self.fwd_thresh = fwd_thresh
        self.bwd_thresh = bwd_thresh
    def fprop(self, input, output):
        output.resize(input.shape)
        thresh_less(input.x, input.x, self.fwd_thresh, output.x)
    def bprop_input (self, input, output):
        thresh_less(output.dx, input.x, self.bwd_thresh, input.dx, True)
    def bbprop_input(self, input, output):
        thresh_less(output.ddx, input.x, self.bwd_thresh, input.ddx, True)

class transfer_double_abs (layers):
    def __init__(self, thresh = 0.0001):
        super(transfer_double_abs, self).__init__(transfer_copy_flipsign(),
                                                  transfer_greater(0., thresh))
