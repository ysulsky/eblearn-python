import scipy as sp
from util import *

class parameter_update (object):
    def __init__(self,
                 eta         = 0.001,
                 max_iters   = 0,
                 decay_l1    = 0,
                 decay_l2    = 0,
                 decay_time  = 1000,
                 inertia     = 0,
                 anneal_amt  = 0.95,
                 anneal_time = 1000,
                 grad_thresh = 0.001):
        vals = dict(locals())
        del vals['self']
        self.__dict__.update(vals)
    
class parameter_forget (object):
    def __init__(self,
                 lin_value    = 1.0,
                 lin_exponent = 2.0):
        vals = dict(locals())
        del vals['self']
        self.__dict__.update(vals)

class parameter (object):
    def __init__(self):
        self.age    = 0
        self.states = []
        self.forget = parameter_forget()

    def merge(self, other):
        if other is None: return
        self.states.extend(other.states)
        other.__dict__.update(self.__dict__)
    
    def append(self, state):
        self.states.append(state)
    
    def size(self):
        return sum(x.size for x in self.states)

    def clear_dx(self):
        for state in self.states: state.dx.fill(0.)

    def clear_ddx(self):
        for state in self.states: state.ddx.fill(0.)

    def clear_ddeltax(self):
        for state in self.states: state.ddeltax.fill(0.)
    
    def set_epsilons(self, v):
        for state in self.states: state.epsilons.fill(v)

    def update(self, arg):
        '''arg is of type parameter_update'''

        eta      = arg.eta
        decay_l1 = arg.decay_l1
        decay_l2 = arg.decay_l2
        inertia  = arg.inertia
        states   = self.states

        if self.age >= arg.decay_time:
            if decay_l2 > 0:
                for state in states: 
                    state.dx += state.x * decay_l2
            if decau_l1 > 0:
                for state in states: 
                    state.dx += sp.sign(state.x) * decay_l1

        if inertia == 0:
            for state in states:
                state.x += state.dx     * state.epsilons * (-eps)
        else:
            for state in states:
                state.deltax = inertia * state.deltax + (1.-inertia) * state.dx
                state.x += state.deltax * state.epsilons * (-eps)

        self.age += 1

    def iter_state_prop(prop):
        def iter(self):
            for state in self.states:
                xs = getattr(state, prop)
                for x in xs.flat: yield x
        return iter

    x        = property(iter_state_prop('x'))
    dx       = property(iter_state_prop('dx'))
    ddx      = property(iter_state_prop('ddx'))
    deltax   = property(iter_state_prop('deltax'))
    ddeltax  = property(iter_state_prop('ddeltax'))
    epsilons = property(iter_state_prop('epsilons'))
