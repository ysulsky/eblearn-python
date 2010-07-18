from module import *

class distance_l2 (module_2_1):
    def fprop(self, input1, input2, energy):
        energy.resize((1,))
        assert (input1.shape == input2.shape)
        energy.x[0] = sqdist(input1.x, input2.x) / (2. * input1.size)

    def bprop_input(self, input1, input2, energy):
        r = (input1.x - input2.x) * (energy.dx[0] / input1.size)
        input1.dx += r
        input2.dx -= r
    
    def bbprop_input(self, input, input2, energy):
        r = energy.dx[0] / input1.size
        input1.ddx += r
        input2.ddx += r

class cross_entropy (module_2_1):
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
        dd1 = input2.x.sum() * softmaxin1 - input2.x
        dd2 = sp.log(softmaxin1)
        input1.ddx += energy.ddx[0] * (dd1 ** 2)
        input2.ddx += energy.ddx[0] * (dd2 ** 2)

