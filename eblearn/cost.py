from module import *

class distance_l2 (no_params, module_2_1):
    def fprop(self, input1, input2, energy):
        energy.resize((1,))
        assert (input1.shape == input2.shape)
        energy.x[0] = sqdist(input1.x, input2.x) / (2. * input1.size)

    def bprop_input(self, input1, input2, energy):
        r = (input1.x - input2.x) * (energy.dx[0] / input1.size)
        input1.dx += r
        input2.dx -= r
    
    def bbprop_input(self, input1, input2, energy):
        r = energy.dx[0] / input1.size
        input1.ddx += r
        input2.ddx += r

class cross_entropy (no_params, module_2_1):
    def fprop(self, input1, input2, energy):
        energy.resize((1,))
        assert (input1.shape == input2.shape)
        expin1 = sp.exp(input1.x)
        softmaxin1 = expin1 * (1.0 / expin1.sum())
        energy.x[0] = sp.dot(-input2.x.ravel(),
                              sp.log(softmaxin1).ravel())

    def bprop_input(self, input1, input2, energy):
        expin1 = sp.exp(input1.x)
        softmaxin1 = expin1 * (1.0 / expin1.sum())
        input1.dx += energy.dx[0] * ((input2.x.sum() * softmaxin1) - input2.x)
        input2.dx -= energy.dx[0] * sp.log(softmaxin1) 
        
    def bbprop_input(self, input1, input2, energy):
        expin1 = sp.exp(input1.x)
        softmaxin1 = expin1 * (1.0 / expin1.sum())
        dd1 = sp.square(input2.x.sum() * softmaxin1 - input2.x)
        dd2 = sp.square(sp.log(softmaxin1))
        input1.ddx += energy.ddx[0] * dd1
        input2.ddx += energy.ddx[0] * dd2

class penalty_l1 (no_params, module_1_1):
    def __init__(self, thresh = 0.0001):
        self.thresh = thresh
    def fprop(self, input, energy):
        energy.resize((1,))
        energy.x[0] = sp.absolute(input.x).sum() / input.size
    def bprop_input (self, input, energy):
        sx = input.x / self.thresh
        sp.trunc(sx, sx)
        sp.sign(sx, sx)
        input.dx += sx * (energy.dx[0] / input.size)
    def bbprop_input(self, input, energy):
        input.ddx += energy.ddx[0]
