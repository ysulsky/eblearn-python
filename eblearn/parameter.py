import scipy as sp
from util import *

class parameter_forget (object):
    def __init__(self,
                 lin_value    = 1.0,
                 lin_exponent = 2.0):
        vals = dict(locals())
        del vals['self']
        self.__dict__.update(vals)


class parameter_update (object):
    def step(self, p):
        raise NotImplementedError()
    def stop_reason(self):
        raise NotImplementedError()
    def reset(self):
        raise NotImplementedError()

class parameter (object):
    cur_id = 1
    
    class update_default (parameter_update):
        def __init__(self, ctor):
            self.ctor = ctor
        def reset(self):
            pass
        def step(self, p):
            print 'Using default update strategy: %s' % self.ctor.__name__
            p.updater = self.ctor()
            return p.updater.step(p)
    
    def __init__(self):
        self.id = parameter.cur_id
        parameter.cur_id += 1
        
        self.age       = 0
        self.states    = []
        self.state_ids = set()
        self.forget    = parameter_forget()
        self.parent    = None
        self.updater   = parameter_update_default_gd

    def reset(self):
        # may be called several times in a row
        if self.age == 0: return
        self.age = 0
        self.updater.reset()

    def stop_reason(self):
        return self.updater.stop_reason()
    
    def merge(self, other):
        if self.id == other.id: return
        assert(other.parent is None)
        other.parent = self
        
        for state in other.states:
            self.append(state)
    
    def append(self, state):
        if state.id not in self.state_ids:
            self.states.append(state)
            self.state_ids.add(state.id)
            if self.parent is not None:
                self.parent.append(state)
    
    def size(self):
        return sum(x.size for x in self.states)

    def clear_dx(self):
        for state in self.states: state.dx.fill(0.)

    def clear_ddx(self):
        for state in self.states: state.ddx.fill(0.)

    def clear_deltax(self):
        for state in self.states: state.deltax.fill(0.)

    def clear_ddeltax(self):
        for state in self.states: state.ddeltax.fill(0.)
    
    def set_epsilon(self, v):
        for state in self.states: state.epsilon.fill(v)

    def compute_epsilon(self, mu):
        for state in self.states: state.epsilon = 1.0 / (state.ddeltax + mu)
    
    def update_deltax(self, knew, kold):
        for state in self.states:
            state.deltax = kold * state.deltax + knew * state.dx
    
    def update_ddeltax(self, knew, kold):
        assert (sp.all(sp.all(state.ddx > -1e-6) for state in self.states))
        for state in self.states:
            state.ddeltax = kold * state.ddeltax + knew * state.ddx
    
    def update(self):
        ret = self.updater.step(self)
        self.age += 1
        return ret

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


class gd_update (parameter_update):
    ''' Gradient-descent parameter update strategy '''
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
        self.reset()
    
    stop_reasons = ['none',
                    'iteration limit reached',
                    'gradient threshold reached']
    
    def stop_reason(self):
        return gd_update.stop_reasons[self.stop_code]

    def reset(self):
        self.stop_code = 0
    
    def step(self, p):
        age         = p.age
        eta         = self.eta
        anneal_time = self.anneal_time
        decay_l1    = self.decay_l1
        decay_l2    = self.decay_l2
        inertia     = self.inertia
        states      = p.states

        if anneal_time > 0 and (age % anneal_time) == 0:
            eta /= 1. + (self.anneal_amt * age / anneal_time)
        
        if age >= self.decay_time:
            if decay_l2 > 0:
                for state in states: 
                    state.dx += state.x * decay_l2
            if decay_l1 > 0:
                for state in states: 
                    state.dx += sp.sign(state.x) * decay_l1
        
        grad = None
        if inertia == 0:
            grad = [state.dx * state.epsilon for state in states]
        else:
            p.update_deltax(inertia, 1.-inertia)
            grad = [state.deltax * state.epsilon for state in states]
        
        grad_norm = max(eta, sqrt(sum(sqmag(g) for g in grad)))
        for (g, state) in zip(grad,states):
            state.x += (-eta / grad_norm) * g
        
        if self.max_iters and age >= self.max_iters: self.stop_code = 1
        if grad_norm < self.grad_thresh:             self.stop_code = 2
        return self.stop_code == 0

parameter_update_default_gd = parameter.update_default(gd_update)


class parameter_container (object):
    def __init__(self, *params):
        self.params = params

    def reset(self):
        for p in self.params: p.reset()

    def stop_reason(self):
        reasons = [p.stop_reason() for p in self.params]
        return '(%s)' % (', '.join(reasons))

    def size(self):
        return sum(p.size() for p in self.params)
    
    def clear_dx(self):
        for p in self.params: p.clear_dx()
    def clear_ddx(self):
        for p in self.params: p.clear_ddx()

    def clear_deltax(self):
        for p in self.params: p.clear_deltax()
    def clear_ddeltax(self):
        for p in self.params: p.clear_ddeltax()

    def update_deltax(self, knew, kold):
        for p in self.params: p.update_deltax(knew, kold)
    def update_ddeltax(self, knew, kold):
        for p in self.params: p.update_ddeltax(knew, kold)        
    
    def set_epsilon(self,v):
        for p in self.params: p.set_epsilon(v)
    def compute_epsilon(self,mu):
        for p in self.params: p.compute_epsilon(mu)

    def update(self):
        ret = False
        for p in self.params: ret = p.update() or ret
        return ret
        
