from eblearn.module  import module_2_1, no_params

import numpy as np

def target_class(targetx):
    # -- no need to assume that targetx is a probability distribution
    # assert ((targetx >= 0).all() and
    #         (targetx.sum() - 1 < 1e-6)), "invalid target"
    return targetx.argmax()

def competing_class(inputx, target_class):
    assert (inputx.size > 1), "can't do classification with one possible class"
    inputx = inputx.copy()
    inputx[target_class] = -np.infty
    return inputx.argmax()


class hinge_loss (no_params, module_2_1):
    def __init__(self, margin = 0.1):
        self.margin = margin
        self.tc = self.cc = -1
    
    def fprop(self, input, target, energy):
        assert (input.shape == target.shape)
        energy.resize((1,))
        
        self.tc = target_class(target.x)
        self.cc = competing_class(input.x, self.tc)
        inputx = input.x.ravel()
        energy.x[0] = max(0, self.margin + inputx[self.cc] - inputx[self.tc])
    
    def bprop_input(self, input, target, energy):
        if energy.x[0] == 0: return
        edx, inputdx = energy.dx[0], input.dx.ravel()
        inputdx[self.cc] += edx
        inputdx[self.tc] -= edx
    
    def bbprop_input(self, input, target, energy):
        if energy.x[0] == 0: return
        eddx, inputddx = energy.ddx[0], input.ddx.ravel()
        inputddx[self.cc] += eddx
        inputddx[self.tc] += eddx
