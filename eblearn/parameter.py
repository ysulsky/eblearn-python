import numpy as np
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
    def iterstats(self):
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
    
    def __init__(self, name=None):
        self.id = parameter.cur_id
        parameter.cur_id += 1
        self._name = ref(name or 'parameter(%d)' % (self.id,))
        
        self.age       = 0
        self.states    = []
        self.state_ids = set()
        self.forget    = parameter_forget()
        self.parents   = set()
        self.updater   = gd_update()
    
    def __getstate__(self):
        parent_refs = [p() for p in self.parents]
        parent_refs = [p for p in parent_refs if p]
        state = dict(self.__dict__)
        state['parents'] = parent_refs
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        parent_refs    = self.parents
        self.parents   = set()
        for p in parent_refs: self._add_parent(p)
    
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

    def _add_parent(self, p):
        self.parents.add(weakref.ref(p, self._clear_parent))
    
    def merge(self, other, keep_updated = True):
        if self is other:                     return
        if self.__weakref__ in other.parents: return
        
        if keep_updated:
            other._add_parent(self)
        
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
            deltax = state.deltax
            mdotc(deltax,   kold, deltax)
            mdotc(state.dx, knew, deltax, True)
    
    def update_ddeltax(self, knew, kold):
        for state in self.states:
            ddeltax = state.ddeltax
            mdotc(ddeltax,   kold, ddeltax)
            mdotc(state.ddx, knew, ddeltax, True)
    
    def update(self):
        self.age += 1
        ret = self.updater.step(self)
        return ret

    def iterstats(self):
        return self.updater.iterstats()
    
    def iter_state_get(prop):
        def xiter(self):
            for state in self.states:
                xs = getattr(state, prop)
                for x in xs.flat: yield x
        return xiter

    def iter_state_set(prop):
        def xiter(self, vals):
            for state in self.states:
                xs = getattr(state, prop)
                vals = iter(vals)
                for i in range(xs.size):
                    xs.flat[i] = vals.next()
        return xiter

    x       = property(iter_state_get('x'),       iter_state_set('x'))
    dx      = property(iter_state_get('dx'),      iter_state_set('dx'))
    ddx     = property(iter_state_get('ddx'),     iter_state_set('ddx'))
    deltax  = property(iter_state_get('deltax'),  iter_state_set('deltax'))
    ddeltax = property(iter_state_get('ddeltax'), iter_state_set('ddeltax'))
    epsilon = property(iter_state_get('epsilon'), iter_state_set('epsilon'))

    def _get_name(self): return self._name.contents
    def _set_name(self, v): self._name.contents = v
    name = property(_get_name, _set_name)


class gd_update (parameter_update):
    ''' Gradient-descent parameter update strategy '''

    def __init__(self,
                 eta         = 0.01,
                 max_iters   = -1,
                 decay_l1    = 0,
                 decay_l2    = 0,
                 decay_time  = 1000,
                 inertia     = 0,
                 anneal_amt  = 0.1,
                 anneal_time = 1000,
                 grad_thresh = 0.0001,
                 norm_grad   = False,
                 thresh_x    = None,
                 debugging   = False):
        vals = dict(locals())
        del vals['self']
        self.__dict__.update(vals)
        self.reset()

    def stop_reason(self):
        if self.stop_code is None: return 'none'
        return self.stop_code

    def reset(self):
        self.stop_code = None
        self.cur_grad_norm = -1.

    def iterstats(self): return {'grad norm': self.cur_grad_norm}

    def _perform_step(self, p, grad, coeff):
        states = p.states
        for (g, state) in zip(grad,states):
            mdotc(g, coeff, state.x, True)
    
    def _step_direction(self, p, dostep = True):
        ''' internal - returns (gradient, step size) '''
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
                    mdotc(        state.x,  decay_l2, state.dx, True)
            if decay_l1 > 0:
                for state in states:
                    mdotc(np.sign(state.x), decay_l1, state.dx, True)
        
        grad = None
        if inertia == 0:
            grad = [state.dx * state.epsilon for state in states]
        else:
            p.update_deltax(inertia, 1.-inertia)
            grad = [state.deltax * state.epsilon for state in states]
        
        grad_norm = sqrt(sum(sumsq(g) for g in grad))
        
        step_coeff = -eta
        if self.norm_grad:
            step_coeff /= max(grad_norm, eta)
        
        if self.max_iters >= 0 and p.age > self.max_iters:
            self.stop_code = 'iteration limit reached'
        if grad_norm < self.grad_thresh:
            self.stop_code = 'gradient threshold reached'

        if self.debugging:
            if min([state.epsilon.min() for state in states]) < 0:
                debug_break('negative epsilon')
            if grad_norm > 50000.0:
                debug_break('huge gradient norm: %g' % grad_norm)

        self.cur_grad_norm = grad_norm

        if dostep:
            self._perform_step(p, grad, step_coeff)
        
        return (grad, step_coeff)
    

    def step(self, p):
        grad, step_coeff = self._step_direction(p, dostep = True)
        states = p.states
        
        if self.thresh_x is not None:
            for state in states:
                thresh_less(state.x, state.x, self.thresh_x, state.x)
        
        return self.stop_code is None


class feval_from_trainer(object):
    def __init__(self, trainer):
        self.trainer         = trainer
        self.trnum           = 0
        self.trage_firsteval = 1
        self.energy          = state((1,))
    def __call__(self):
        trainer = self.trainer
            
        if trainer.train_num != self.trnum:
            self.trnum = trainer.train_num
            self.trage_firsteval = 1

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
            
            SEE: feval_from_trainer '''
        
        self.feval                = feval
        self.max_line_steps       = max_line_steps
        self.quiet                = quiet

        if 'eta' not in kwargs: kwargs['eta'] = 0.5
        super(gd_linesearch_update, self).__init__(**kwargs)
    
    def reset(self):
        self.cur_num_steps        = -1
        super(gd_linesearch_update, self).reset()

    def iterstats(self):
        r = super(gd_linesearch_update, self).iterstats()
        r['line search steps'] = self.cur_num_steps
        return r
    
    def _step_direction(self, p, dostep=True):
        grad, step_coeff = \
              super(gd_linesearch_update, self)._step_direction(p, False)
        
        feval  = self.feval
        states = p.states
        bup    = p.backup()
        stop   = self.max_line_steps
        step   = 0
        
        cur_energy = feval()
        new_energy = np.infty
        
        while step != stop:
            step += 1
            self._perform_step(p, grad, step_coeff)
            new_energy = feval()
            
            if new_energy < cur_energy:
                break
            
            step_coeff /= 2.
            p.restore(bup)
            
        
        self.cur_num_steps = step
        if new_energy >= cur_energy:
            self.stop_code = 'line search failed'
        
        if not self.quiet:
            print 'linesearch: stopped after %d iterations because: %s' % \
                  (step, ('energy decreased' if new_energy < cur_energy else
                          'iteration limit reached'))
        
        if not dostep:
            p.restore(bup)
        
        return grad, step_coeff


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
        reasons = [p.name+': '+p.stop_reason() for p in self.params]
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
    
    def iterstats(self):
        r = {}
        for p in self.params:
            name, stats = p.name, p.iterstats()
            for k, v in stats.items():
                r[name + ' ' + k] = v
        return r

    def iter_param_get(prop):
        def xiter(self):
            for p in self.params:
                for x in getattr(p, prop):
                    yield x
        return xiter

    def iter_param_set(prop):
        def xiter(self, vals):
            vals = iter(vals)
            for p in self.params:
                setattr(p, prop, vals)
        return xiter
    
    x       = property(iter_param_get('x'),       iter_param_set('x'))
    dx      = property(iter_param_get('dx'),      iter_param_set('dx'))
    ddx     = property(iter_param_get('ddx'),     iter_param_set('ddx'))
    deltax  = property(iter_param_get('deltax'),  iter_param_set('deltax'))
    ddeltax = property(iter_param_get('ddeltax'), iter_param_set('ddeltax'))
    epsilon = property(iter_param_get('epsilon'), iter_param_set('epsilon'))

    def _get_name(self):
        return '(%s)'%(', '.join([p.name for p in self.params]),)
    
    name    = property(_get_name)
