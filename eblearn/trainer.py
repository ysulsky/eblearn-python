from eblearn import *

import time

class eb_trainer (object):
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
                 do_normalization  = False):
        vals = dict(locals())
        del vals['self']
        self.__dict__.update(vals)

        if not parameter: self.parameter = machine.parameter
        
        assert (self.parameter and self.parameter.size() > 0)
        assert (ds_train.size() > 0) 

        if ds_valid is not None and ds_valid.size() <= 0:
            self.ds_valid = None
        if self.ds_valid is None: 
            self.valid_interval = 0
        
        self.age    = 0
        self.input  = state(())
        self.target = state(())
        self.energy = state((1,))
        self.energy.dx[0]  = 1.
        self.energy.ddx[0] = 1.

        self.msg = self.make_msg()
        self.clear_log()

    def clear_log(self):
        self.train_loss = []
        self.valid_loss = []

    def make_msg(self):
        prev_age = [-1]
        prefix_width = 20
        def prefix_age(age):
            s_age = str(age)
            dots  = '.' * (prefix_width - 6 - len(s_age))
            return 'Age %s%s: ' % (s_age, dots)
        def prefix_noage():
            return '.' * (prefix_width - 1) + ' '
        def msg(s):
            prefix = prefix_noage()
            if self.age != prev_age[0]:
                prefix = prefix_age(self.age)
                prev_age[0] = self.age
            print prefix + s
        return msg

    def train(self, niter):
        self.age = 0
        self.clear_log()
        self.ds_train.seek(0)
        self.machine.forget()
        self.train_online(niter)
        self.msg = self.make_msg()
        
    def train_online(self, niter):
        print 'Starting training on %s (%d iterations)'%(time.asctime(),niter)

        msg             = self.msg
        hess_interval   = self.hess_interval
        report_interval = self.report_interval
        valid_interval  = self.valid_interval
        valid_err_tol   = self.valid_err_tol
        backup_interval = self.backup_interval
        prev_vld_loss   = None

        if hess_interval <= 0:
            self.parameter.set_epsilon(1.)

        for i in xrange(niter):
            age = self.age
            
            if hess_interval > 0 and (age % hess_interval) == 0:
                self.compute_diag_hessian()
                
            tr_loss = self.train_sample()
            
            if self.keep_log:
                self.train_loss.append((tr_loss))
                if      report_interval > 0 and \
                        age > 0 and (age % report_interval) == 0:
                    avloss = sp.mean(self.train_loss[-report_interval:])
                    msg('av. loss = %g' % avloss)

            if valid_interval > 0 and age > 0 and (age % valid_interval) == 0:
                vld_loss = self.validate()
                if self.keep_log: self.valid_loss.append((age, vld_loss))
                msg('validation loss = %g' % vld_loss)

                if valid_err_tol >= 0 and prev_vld_loss is not None:
                    if vld_loss - prev_vld_loss > valid_err_tol * prev_vld_loss:
                        print 'Stopping due to increased validation loss'
                        break

                prev_vld_loss = vld_loss

            if backup_interval > 0 and age > 0 and (age % backup_interval) == 0:
                fname = '%s/%s_%d.obj' % (self.backup_location, 
                                          self.backup_name,
                                          age // backup_interval)
                pickle.dump(self.machine, open(fname, 'wb'))
        
            self.ds_train.next()
            self.age += 1
        
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
        self.parameter.update()
        if self.do_normalization:
            machine.normalize()
        
        return energy.x[0]

    def train_sample_bbprop(self):
        machine, energy = self.machine, self.energy
        knew = 1.0 / self.hess_iter
        kold = 1.0
        
        self._fprop_bprop()
        self.parameter.clear_ddx()
        machine.bbprop(self.input, self.target, energy)
        self.parameter.update_ddeltax(knew, kold)
        
        return energy.x[0]

    def compute_diag_hessian(self):
        self.msg('computing diagonal Hessian approximation')
        start_pos = self.ds_train.tell()
        for i in xrange(self.hess_iter):
            self.train_sample_bbprop()
            self.ds_train.next()
        self.ds_train.seek(start_pos)
        self.parameter.compute_epsilon(self.hess_mu)

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

