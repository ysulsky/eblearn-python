from eblearn.state import state
from eblearn.util  import abuffer, save_object

import numpy as np
import time

class eb_trainer (object):
    
    class reporter (object):
        def __init__(self, trainer):
            self.trainer = trainer
            self.prev_age      = 0
            self.prefix_width  = 15
        def __call__(self, msg):
            if self.trainer.quiet: return
            def prefix_age(age):
                s_age = str(age)
                dots  = '.' * (self.prefix_width - 6 - len(s_age))
                return 'Age %s%s: ' % (s_age, dots)
            def prefix_noage():
                return '.' * (self.prefix_width - 1) + ' '
            prefix = prefix_noage()
            if self.trainer.age != self.prev_age:
                prefix = prefix_age(self.trainer.age)
                self.prev_age = self.trainer.age
            print prefix + msg
    
    def __init__(self, parameter, machine, ds_train,
                 ds_valid          = None, 
                 valid_interval    = 2000,
                 valid_iters       = -1,
                 valid_err_tol     = 0.1,
                 report_interval   = 2000,
                 hess_interval     = 2000,
                 hess_iters        = 500,
                 hess_mu           = 0.02,
                 backup_interval   = -1,
                 backup_location   = '.',
                 keep_log          = True,
                 do_normalization  = False,
                 complete_training = True,
                 quiet             = False,
                 auto_forget       = True,
                 verbose           = False):
        vals = dict(locals())
        del vals['self']
        self.__dict__.update(vals)
        
        if not backup_location:
            self.backup_interval = -1
        
        if quiet: self.verbose = False
        
        assert (parameter and parameter.size() > 0)
        assert (ds_train.size() > 0) 
        
        if ds_valid is not None and ds_valid.size() <= 0:
            self.ds_valid = None
        if self.ds_valid is None: 
            self.valid_interval = 0
        
        self.train_num     = 0
        self.age           = 0
        self.input         = state()
        self.target        = state()
        self.energy        = state((1,))
        self.energy.dx[0]  = 1
        self.energy.ddx[0] = 0
        
        self.msg = eb_trainer.reporter(self)
        self.clear_log()
        self.clear_stop()

        self._tracked_stats = []
        self.track_stats(machine)

    def clear_log(self):
        self.valid_loss     = None
        self.train_stats    = None
        if self.keep_log:
            if self.valid_interval > 0:
                self.valid_loss = abuffer((50, 2))
            self.train_stats = {'training loss': abuffer()}
        self.last_stop      = None
        self.last_valid     = np.infty

    def track_stats(self, mod):
        if not self.keep_log: return
        stats = mod.get_stats()
        if self.age > 0:
            for k in stats:
                stat_name = mod._name.contents + ' ' + k
                self.train_stats[stat_name] = [np.nan] * self.age
        self._tracked_stats.append((mod._name, stats))

    def clear_stop(self):
        self.stop_code = None
    
    def stop_reason(self):
        return self.stop_code
    
    def train(self, niters = -1):
        self.train_num += 1
        self.age = 0
        self.msg = eb_trainer.reporter(self)
        self.clear_log()
        self.clear_stop()
        self.ds_train.seek(0)
        self.parameter.reset()
        if self.auto_forget:
            self.machine.forget()
        self.train_online(niters)
    
    def log_stats(self):
        if not self.keep_log: return
        
        train_stats = self.train_stats
        def append_stats(stats, prefix = None):
            for k, v in stats.items():
                if prefix is not None: k = prefix+k
                log = train_stats.get(k)
                if log is None: train_stats[k] = log = abuffer()
                log.append(v)
        
        train_stats['training loss'].append(self.energy.x[0])
        append_stats(self.parameter.iterstats())
        for (nameref, stats) in self._tracked_stats:
            append_stats(stats, nameref.contents + ' ')
    
    def average_stats(self, num = None, stats = None):
        if not self.keep_log: return {}
        ret = {}
        if stats is None: stats = self.train_stats.iterkeys()
        for stat in stats:
            log = self.train_stats.get(stat)
            if log is None: continue
            assert(len(log) == self.age)
            if num is not None: log = log[-num:]
            ret['avg. '+stat] = np.mean(log, 0)
        return ret
    
    def backup_machine(self, dest = None):
        if dest is None:
            dest = '%s/%s_%d.obj' % (self.backup_location,
                                     self.machine.name, self.age)
        save_object(self.machine, dest)
    
    def backup_stats(self, dest = None):
        if dest is None:
            dest = '%s/%s_%d.obj' % (self.backup_location,
                                     'train_stats', self.age)
        save_object(self.train_stats, dest)
    
    def report_stats(self):
        msg   = self.msg
        stats = None if self.verbose else ('training loss',)
        avg_stats = self.average_stats(self.report_interval, stats)
        for field in sorted(avg_stats):
            msg('%s = %g' % (field, avg_stats[field]))
    
    def train_online(self, niters = -1):
        if not self.quiet:
            print 'Starting training on %s%s' % \
                  (time.asctime(),
                   ' (max %d iterations)' % (niters,) if niters >= 0 else '')
        
        parameter       = self.parameter
        
        age             = self.age
        msg             = self.msg
        hess_interval   = self.hess_interval
        report_interval = self.report_interval
        valid_interval  = self.valid_interval
        backup_interval = self.backup_interval
        keep_training   = True
        
        if hess_interval <= 0: parameter.set_epsilon(1.)
        else:                  self.compute_diag_hessian()
        
        stop_age     = age + niters
        min_finished = 0
        if self.complete_training:
            min_finished = self.ds_train.size() - self.ds_train.tell()
        
        while age != stop_age and (keep_training or age < min_finished):

            if not keep_training:
                reason = self.stop_reason()
                if self.last_stop is None:
                    msg('%s, but continuing with remaining samples' % (reason,))
                self.last_stop = reason
                self.clear_stop()
                keep_training = True
            
            self.age += 1
            age = self.age
            
            if hess_interval > 0 and (age % hess_interval) == 0:
                self.compute_diag_hessian()
            
            keep_training = self.train_sample()

            self.log_stats()
            
            if report_interval > 0 and (age % report_interval) == 0:
                self.report_stats()
            
            if valid_interval > 0 and (age % valid_interval) == 0:
                valid_ok, valid_loss = self.validate()
                keep_training = keep_training and valid_ok
                msg('validation loss = %g' % (valid_loss,))
            
            if backup_interval > 0 and (age % backup_interval) == 0:
                self.backup_machine()
                self.backup_stats()
            
            self.ds_train.next()
        
        
        if not keep_training:
            msg('stopping because %s' % (self.stop_reason(),))
            if self.verbose:
                for field in sorted(self.train_stats):
                    lastval = self.train_stats[field][-1]
                    msg('last iteration: %s = %g' % (field, lastval))
        
        if age == stop_age:
            msg('completed %d iterations' % (niters,))
        
        if not self.quiet:
            print 'Ended training on %s'%(time.asctime())
    
    def _fprop_bprop(self):
        machine, energy = self.machine, self.energy
        input, target   = self.input, self.target
        
        self.ds_train.fprop(input, target)
        machine.fprop(input, target, energy)
        self.parameter.clear_dx()
        machine.bprop(input, target, energy)
        
    def train_sample(self):
        machine, energy = self.machine, self.energy
        
        self._fprop_bprop()
        keep_training = self.parameter.update()
        if self.do_normalization:
            machine.normalize()
        
        if not keep_training:
            self.stop_code = self.parameter.stop_reason()
        
        return keep_training

    def train_sample_bbprop(self):
        machine, energy = self.machine, self.energy
        knew = 1.0 / self.hess_iters
        kold = 1.0
        
        self._fprop_bprop()
        self.parameter.clear_ddx()
        machine.bbprop(self.input, self.target, energy)
        self.parameter.update_ddeltax(knew, kold)
    
    def compute_diag_hessian(self):
        self.msg('computing diagonal Hessian approximation (%d iterations)'\
                 % (self.hess_iters,))
        start_pos = self.ds_train.tell()
        for i in xrange(self.hess_iters):
            self.train_sample_bbprop()
            self.ds_train.next()
        self.ds_train.seek(start_pos)
        self.parameter.compute_epsilon(self.hess_mu)
        if self.verbose:
            self.msg('avg. epsilon = %g' % np.mean(self.parameter.epsilon))
        else:
            self.msg('done.')
    
    def validate(self):
        ds_valid = self.ds_valid
        
        n = self.valid_iters
        if n <= 0: n = ds_valid.size()
        
        machine, energy = self.machine, self.energy
        input, target   = self.input, self.target
        
        tot_err = 0.0
        for i in xrange(n):
            ds_valid.fprop(input, target)
            machine.fprop(input, target, energy)
            tot_err += energy.x[0]
            ds_valid.next()
        
        tol, last_err = self.valid_err_tol, self.last_valid
        err = tot_err / n
        
        good = True
        if tol >= 0 and err - last_err > tol * last_err:
            self.stop_code = 'validation loss increased'
            good = False
        
        if self.valid_loss is not None:
            self.valid_loss.append((self.age, err))
        
        self.last_valid = err
        return good, err

# for pickling
reporter = eb_trainer.reporter
