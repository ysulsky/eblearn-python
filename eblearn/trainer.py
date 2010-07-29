from eblearn import *
from util import *

import pickle, time

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
                 valid_iter        = -1,
                 valid_err_tol     = 0.1,
                 report_interval   = 2000,
                 hess_interval     = 2000,
                 hess_iter         = 500,
                 hess_mu           = 0.02,
                 backup_interval   = 0,
                 backup_location   = '.',
                 backup_name       = 'machine',
                 keep_log          = True,
                 do_normalization  = False,
                 quiet             = False,
                 auto_forget       = True,
                 verbose           = False):
        vals = dict(locals())
        del vals['self']
        self.__dict__.update(vals)
        
        if not backup_location:
            self.backup_interval = 0
        
        if quiet: self.verbose = False
        
        assert (parameter and parameter.size() > 0)
        assert (ds_train.size() > 0) 
        
        if ds_valid is not None and ds_valid.size() <= 0:
            self.ds_valid = None
        if self.ds_valid is None: 
            self.valid_interval = 0
        
        self.train_num     = 0
        self.age           = 0
        self.input         = state(())
        self.target        = state(())
        self.energy        = state((1,))
        self.energy.dx[0]  = 1.
        self.energy.ddx[0] = 1.
        
        self.msg = eb_trainer.reporter(self)
        self.clear_log()

        self._tracked_stats = []
        self.track_stats(machine)

    def clear_log(self):
        self.valid_loss     = None
        self.train_stats    = None
        if self.keep_log:
            if self.valid_interval > 0:
                self.valid_loss  = abuffer((1,2))
            self.train_stats  = {'training loss': abuffer()}

    def track_stats(self, mod):
        if not self.keep_log: return
        stats = mod.get_stats()
        if self.age > 0:
            for k in stats:
                stat_name = mod._name.contents + ' ' + k
                self.train_stats[stat_name] = [np.nan] * self.age
        self._tracked_stats.append((mod._name, stats))
    
    def train(self, maxiter = 0):
        self.train_num += 1
        self.age = 0
        self.msg = eb_trainer.reporter(self)
        self.clear_log()
        self.ds_train.seek(0)
        self.parameter.reset()
        if self.auto_forget:
            self.machine.forget()
        self.train_online(maxiter)
    
    def append_stats(self, stats, prefix=''):
        if not self.keep_log: return
        stat_logs = self.train_stats
        for k, v  in stats.items():
            k = prefix + k
            log = stat_logs.get(k)
            if log is None:
                stat_logs[k] = log = abuffer()
            log.append(v)
    
    def average_stats(self, num = None):
        if not self.keep_log: return {}
        ret = {}
        for k, log in self.train_stats.iteritems():
            assert(len(log) == self.age)
            if num is not None: log = log[-num:]
            ret['avg. '+k] = np.mean(log, 0)
        return ret
    
    def train_online(self, maxiter = 0):
        if not self.quiet:
            print 'Starting training on %s%s' % \
                  (time.asctime(),
                   ' (max %d iterations)' % maxiter if maxiter else '')

        parameter       = self.parameter

        msg             = self.msg
        hess_interval   = self.hess_interval
        report_interval = self.report_interval
        valid_interval  = self.valid_interval
        valid_err_tol   = self.valid_err_tol
        backup_interval = self.backup_interval
        prev_vld_loss   = None
        keep_training   = True
        stop_condition  = None

        training_loss_log = None
        if self.keep_log:
            training_loss_log = self.train_stats['training loss']
        
        if hess_interval <= 0: parameter.set_epsilon(1.)
        else:                  self.compute_diag_hessian()
        
        stop_age = self.age + maxiter
        
        while True:
            self.age += 1
            age = self.age
            
            if hess_interval > 0 and (age % hess_interval) == 0:
                self.compute_diag_hessian()
            
            keep_training = self.train_sample()
            if not keep_training:
                stop_condition = parameter.stop_reason()
            
            if self.keep_log:
                training_loss_log.append(self.energy.x[0])
                self.append_stats(parameter.iterstats())
                for (nameref, stats) in self._tracked_stats:
                    self.append_stats(stats, prefix=nameref.contents+' ')
                
                if report_interval > 0 and (age % report_interval) == 0:
                    if self.verbose:
                        avg_stats = self.average_stats(report_interval)
                        for field in sorted(avg_stats):
                            msg('%s = %g' % (field, avg_stats[field]))
                    else:
                        avloss = np.mean(training_loss_log[-report_interval:])
                        msg('avg. training loss = %g' % avloss)
            
            if valid_interval > 0 and (age % valid_interval) == 0:
                vld_loss = self.validate()
                if self.valid_loss is not None:
                    self.valid_loss.append((age, vld_loss))
                msg('validation loss = %g' % vld_loss)

                if valid_err_tol >= 0 and prev_vld_loss is not None:
                    if vld_loss - prev_vld_loss > valid_err_tol * prev_vld_loss:
                        msg('stopping due to increased validation loss')
                        keep_training = False
                        
                prev_vld_loss = vld_loss

            if backup_interval > 0 and (age % backup_interval) == 0:
                fname = '%s/%s_%d.obj' % (self.backup_location, 
                                          self.backup_name,
                                          age // backup_interval)
                pickle.dump(self.machine, open(fname, 'wb'),
                            protocol = pickle.HIGHEST_PROTOCOL)

            if stop_condition is not None:
                msg('stopping because condition was reached: %s' % (stop_condition,))
                if self.verbose:
                    for field in sorted(self.train_stats):
                        lastval = self.train_stats[field][-1]
                        msg('last iteration: %s = %g' % (field, lastval))
            
            if age == stop_age:
                msg('stopping after %d iterations' % maxiter)
                keep_training = False
            
            self.ds_train.next()
            
            if not keep_training: break
        
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
        
        return keep_training

    def train_sample_bbprop(self):
        machine, energy = self.machine, self.energy
        knew = 1.0 / self.hess_iter
        kold = 1.0
        
        self._fprop_bprop()
        self.parameter.clear_ddx()
        machine.bbprop(self.input, self.target, energy)
        self.parameter.update_ddeltax(knew, kold)
    
    def compute_diag_hessian(self):
        self.msg('computing diagonal Hessian approximation')
        start_pos = self.ds_train.tell()
        for i in xrange(self.hess_iter):
            self.train_sample_bbprop()
            self.ds_train.next()
        self.ds_train.seek(start_pos)
        self.parameter.compute_epsilon(self.hess_mu)
        eps = None
        if self.verbose:
            eps = np.fromiter(self.parameter.epsilon, rtype)
            self.msg('avg. epsilon = %g' % eps.mean())
        else:
            self.msg('done.')
    
    def validate(self):
        assert (self.valid_interval > 0)
        
        ds_valid = self.ds_valid

        n = self.valid_iter
        if n <= 0: n = ds_valid.size()
        
        machine, energy = self.machine, self.energy
        input, target   = self.input, self.target
        
        tot_err = 0.0
        for i in xrange(n):
            ds_valid.fprop(input, target)
            machine.fprop(input, target, energy)
            tot_err += energy.x[0]
            ds_valid.next()

        return tot_err / n

# for pickling
reporter = eb_trainer.reporter
