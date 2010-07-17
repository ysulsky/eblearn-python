from scipy import ndarray
from util import *

class state (object):

    def __init__ (self, shape):
        self.x    = zeros(shape)
        self._dx  = None
        self._ddx = None

        # only used for updating parameter

        self._gradient = None
        self._deltax   = None
        self._ddeltax  = None
        self._epsilons = None

    def resize(self, shape):
        self.x.resize(shape)
        if self._dx       is not None: self._dx.resize(shape)
        if self._ddx      is not None: self._ddx.resize(shape)
        if self._gradient is not None: self._gradient.resize(shape)
        if self._deltax   is not None: self._deltax.resize(shape)
        if self._ddeltax  is not None: self._ddeltax.resize(shape)
        if self._epsilons is not None: self._epsilons.resize(shape)

    shape = property(lambda self: self.x.shape)
    size  = property(lambda self: self.x.size)

    def get_dx(self):
        if self._dx is None:
            self._dx = zeros(self.shape)
        return self._dx
    def set_dx(self, val): self._dx = val

    def get_ddx(self):
        if self._ddx is None:
            self._ddx = zeros(self.shape)
        return self._ddx
    def set_ddx(self, val): self._ddx = val

    dx  = property(get_dx,  set_dx)
    ddx = property(get_ddx, set_ddx)    

    def get_gradient(self):
        if self._gradient is None:
            self._gradient = zeros(self.shape)
        return self._gradient

    def get_deltax(self):
        if self._deltax is None:
            self._deltax = zeros(self.shape)
        return self._deltax

    def get_ddeltax(self):
        if self._ddeltax is None:
            self._ddeltax = zeros(self.shape)
        return self._ddeltax

    def get_epsilons(self):
        if self._epsilons is None:
            self._epsilons = ones(self.shape)
        return self._epsilons

    gradient = property(get_gradient)
    deltax   = property(get_deltax)
    ddeltax  = property(get_ddeltax)
    epsilons = property(get_epsilons)
