import scipy as sp
from util import *

class parameter_update (object):
    def __init__(self,
                 eta         = 0.01,
                 max_iters   = 0,
                 decay_l1    = 0,
                 decay_l2    = 0,
                 decay_time  = 1000,
                 inertia     = 0,
                 anneal_amt  = 0.1,
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
        self.indep  = False
        self.states = []
        self.forget = parameter_forget()

    def reset(self):
        # may be called several times in a row
        if self.age == 0: return
        self.age = 0

    def merge(self, other):
        if other is None: return
        self.states.extend(other.states)
        if not other.indep:
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
    
    def set_epsilon(self, v):
        for state in self.states: state.epsilon.fill(v)

    def compute_epsilon(self, mu):
        for state in self.states: state.epsilon = 1.0 / (state.ddeltax + mu)
    
    def update_ddeltax(self, knew, kold):
        for state in self.states:
            state.ddeltax = kold * state.ddeltax + knew * state.ddx
    
    def update(self, arg):
        '''arg is of type parameter_update'''

        age         = self.age
        eta         = arg.eta
        anneal_time = arg.anneal_time
        decay_l1    = arg.decay_l1
        decay_l2    = arg.decay_l2
        inertia     = arg.inertia
        states      = self.states

        if anneal_time > 0 and (age % anneal_time) == 0:
            eta /= 1. + (arg.anneal_amt * age / anneal_time)
        
        if age >= arg.decay_time:
            if decay_l2 > 0:
                for state in states: 
                    state.dx += state.x * decay_l2
            if decay_l1 > 0:
                for state in states: 
                    state.dx += sp.sign(state.x) * decay_l1

        if inertia == 0:
            for state in states:
                state.x += state.dx     * state.epsilon * (-eta)
        else:
            for state in states:
                state.deltax = inertia * state.deltax + (1.-inertia) * state.dx
                state.x += state.deltax * state.epsilon * (-eta)

        self.age += 1

    def iter_state_prop(prop):
        def iter(self):
            for state in self.states:
                xs = getattr(state, prop)
                for x in xs.flat: yield x
        return iter

    x       = property(iter_state_prop('x'))
    dx      = property(iter_state_prop('dx'))
    ddx     = property(iter_state_prop('ddx'))
    deltax  = property(iter_state_prop('deltax'))
    ddeltax = property(iter_state_prop('ddeltax'))
    epsilon = property(iter_state_prop('epsilon'))
