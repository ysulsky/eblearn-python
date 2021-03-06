from eblearn.util    import ones, zeros
from eblearn.vecmath import clear

class state (object):

    cur_id = 1

    def __init__ (self, shape=(1,)):
        self.id   = state.cur_id
        state.cur_id += 1
        
        self.x    = zeros(shape)
        self._dx  = None
        self._ddx = None

        # only used for updating parameter

        self._gradient = None
        self._deltax   = None
        self._ddeltax  = None
        self._epsilon  = None

    def clear_dx(self):
        if self._dx is not None: clear(self._dx)

    def clear_ddx(self):
        if self._ddx is not None: clear(self._ddx)

    def resize(self, shape):
        self.x.resize(shape)
        if self._dx       is not None: self._dx.resize(shape)
        if self._ddx      is not None: self._ddx.resize(shape)
        if self._gradient is not None: self._gradient.resize(shape)
        if self._deltax   is not None: self._deltax.resize(shape)
        if self._ddeltax  is not None: self._ddeltax.resize(shape)
        if self._epsilon  is not None: self._epsilon.resize(shape)


    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        ret = state(())
        ret.x = self.x[i]
        if self._dx       is not None: ret._dx       = self._dx[i]
        if self._ddx      is not None: ret._ddx      = self._ddx[i]
        if self._gradient is not None: ret._gradient = self._gradient[i]
        if self._deltax   is not None: ret._deltax   = self._deltax[i]
        if self._ddeltax  is not None: ret._ddeltax  = self._ddeltax[i]
        if self._epsilon  is not None: ret._epsilon  = self._epsilon[i]
    
    ndim  = property(lambda self: self.x.ndim)
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

    def get_gradient(self):
        if self._gradient is None:
            self._gradient = zeros(self.shape)
        return self._gradient
    def set_gradient(self, val): self._gradient = val

    def get_deltax(self):
        if self._deltax is None:
            self._deltax = zeros(self.shape)
        return self._deltax
    def set_deltax(self, val): self._deltax = val

    def get_ddeltax(self):
        if self._ddeltax is None:
            self._ddeltax = zeros(self.shape)
        return self._ddeltax
    def set_ddeltax(self, val): self._ddeltax = val

    def get_epsilon(self):
        if self._epsilon is None:
            self._epsilon = ones(self.shape)
        return self._epsilon
    def set_epsilon(self, val): self._epsilon = val

    dx       = property(get_dx,       set_dx)
    ddx      = property(get_ddx,      set_ddx)    
    gradient = property(get_gradient, set_gradient)
    deltax   = property(get_deltax,   set_deltax)
    ddeltax  = property(get_ddeltax,  set_ddeltax)
    epsilon  = property(get_epsilon,  set_epsilon)
