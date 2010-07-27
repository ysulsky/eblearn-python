import scipy as sp
from state import *
from util  import *
import weakref

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
        self.parents   = set()
        self.updater   = parameter_update_default_gd
    
    def reset(self):
        # may be called several times in a row
        if self.age == 0: return
        self.age = 0
        self.updater.reset()

    def backup(self):
        ''' returns an object to be used with restore
            does not save gradients '''
        return [state.x.copy() for state in self.states]

    def restore(self, backup):
        for state, src  in zip(self.states, backup):
            state.x[:] = src

    def stop_reason(self):
        return self.updater.stop_reason()

    def _clear_parent(self, p):
        self.parents.remove(p)
    
    def merge(self, other, keep_updated = True):
        if self is other:                     return
        if self.__weakref__ in other.parents: return
        
        if keep_updated:
            other.parents.add(weakref.ref(self, other._clear_parent))
        
        for state in other.states:
            self.append(state)
    
    def append(self, state):
        if state.id not in self.state_ids:
            self.states.append(state)
            self.state_ids.add(state.id)
            for p in self.parents:
                p = p()
                if p is None: continue
                p.append(state)
    
    def size(self):
        return sum(x.size for x in self.states)

    def clear_dx(self):
        for state in self.states: clear(state.dx)

    def clear_ddx(self):
        for state in self.states: clear(state.ddx)

    def clear_deltax(self):
        for state in self.states: clear(state.deltax)

    def clear_ddeltax(self):
        for state in self.states: clear(state.ddeltax)
    
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

# for pickling
update_default = parameter.update_default


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
                 grad_thresh = 0.001,
                 norm_grad   = False,
                 thresh_x    = None):
        vals = dict(locals())
        del vals['self']
        self.__dict__.update(vals)
        self.reset()

    def stop_reason(self):
        if self.stop_code is None: return 'none'
        return self.stop_code

    def reset(self):
        self.stop_code = None

    def _step_direction(self, p):
        ''' internal - returns (gradient, |gradient|, and step) '''
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
        
        grad_norm = sqrt(sum(sqmag(g) for g in grad))
        
        step_coeff = -eta
        if self.norm_grad:
            step_coeff /= max(grad_norm, eta)
        
        if self.max_iters and p.age >= self.max_iters:
            self.stop_code = 'iteration limit reached'
        if grad_norm < self.grad_thresh:
            self.stop_code = 'gradient threshold reached'
        
        return (grad, grad_norm, step_coeff)
    

    def step(self, p):
        grad, grad_norm, step_coeff = self._step_direction(p)
        states = p.states
        
        for (g, state) in zip(grad,states):
            state.x += step_coeff * g

        if self.thresh_x is not None:
            for state in states:
                thresh_less(state.x, state.x, self.thresh_x, state.x)
        
        return self.stop_code is None

parameter_update_default_gd = parameter.update_default(gd_update)


class feval_from_trainer(object):
    def __init__(self, trainer):
        self.trainer         = trainer
        self.trnum           = 0
        self.trage_firsteval = 0
        self.energy          = state((1,))
    def __call__(self):
        trainer = self.trainer
            
        if trainer.train_num != self.trnum:
            self.trnum = trainer.train_num
            self.trage_firsteval = 0

        if trainer.age == self.trage_firsteval:
            # first one's free
            self.trage_firsteval = trainer.age + 1
            return trainer.energy.x[0]

        trainer.machine.fprop(trainer.input, trainer.target, self.energy)
        return self.energy.x[0]

class gd_linesearch_update (gd_update):
    def __init__(self, feval, max_line_steps=10, quiet=True, **kwargs):
        ''' Gradient-descent parameter update strategy, performing
            a line-search to select the step size

            feval: () -> energy
            see: gd_linesearch_update.feval_from_trainer '''
        
        self.feval                = feval
        self.max_line_steps       = max_line_steps
        self.quiet                = quiet
        self.linesearch_stop_code = None

        if 'eta' not in kwargs: kwargs['eta'] = 0.5
        super(gd_linesearch_update, self).__init__(**kwargs)
    
    def reset(self):
        self.linesearch_stop_code = None
        super(gd_linesearch_update, self).reset()

    def _step_direction(self, p):
        grad, grad_norm, step_coeff = \
              super(gd_linesearch_update, self)._step_direction(p)

        feval  = self.feval
        states = p.states
        bup    = p.backup()
        stop   = self.max_line_steps - 1
        step   = 0

        cur_energy = feval()
        new_energy = sp.infty
        
        while new_energy > cur_energy and step != stop:
            for (g, state) in zip(grad,states):
                state.x += step_coeff * g
            
            new_energy = feval()

            step_coeff /= 2.
            p.restore(bup)
            step += 1

        if step == stop:
            self.linesearch_stop_code = 'iteration limit reached'
        else:
            self.linesearch_stop_code = 'energy decreased'
        
        if not self.quiet:
            print 'linesearch: stopped after %d iterations because: %s' % \
                  (step, self.linesearch_stop_code)
        
        return grad, grad_norm, step_coeff



class parameter_container (object):
    def __init__(self, *params):
        self.params = params

    def reset(self):
        for p in self.params: p.reset()

    def backup(self):
        return [p.backup() for p in self.params]

    def restore(self, backup):
        for p in self.params: p.restore()

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
        
