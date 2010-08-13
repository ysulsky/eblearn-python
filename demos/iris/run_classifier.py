import eblearn as eb
import numpy   as np

from params import *

######################################################################
# DATASOURCE

lines = [x.split(',') for x in open(data_file).read().split('\n') if x]

all_classes = sorted(set([x[-1] for x in lines]))
all_targets = np.diag(np.ones(len(all_classes)))

class_target = dict(zip(all_classes, all_targets))

inputs  = [map(float, x[:-1])  for x in lines]
targets = [class_target[x[-1]] for x in lines]

ds_train = eb.dsource_sup(inputs[0::2], targets[0::2])
ds_test  = eb.dsource_sup(inputs[1::2], targets[1::2])

ds_train.normalize()
ds_test.bias, ds_test.coeff = ds_train.bias, ds_train.coeff

shape_in, shape_out = ds_train.shape()

######################################################################
# CLASSIFIER

if   classifier_arch == 1: # linear
    classifier = eb.layers(eb.linear(shape_in, shape_out),
                           eb.bias_module(shape_out))
elif classifier_arch == 2: # two-layer mlp
    classifier = eb.layers(eb.linear(shape_in, hidden_units),
                           eb.bias_module(hidden_units),
                           eb.transfer_tanh(),
                           eb.linear(hidden_units, shape_out),
                           eb.bias_module(shape_out))
else:
    msg = 'run_classifier: classifier_arch = %d' % (classifier_arch,)
    raise NotImplementedError(msg)

######################################################################
# COST

if   cost_type == 1: # cross-entropy
    cost = eb.cross_entropy()
elif cost_type == 2: # distance
    cost = eb.distance_l2()
else:
    raise NotImplementedError('run_classifier: cost_type = %d' % (cost_type,))

######################################################################
# TRAINING

machine   = eb.ebm_2(classifier, cost)
parameter = machine.parameter

parameter.updater = eb.gd_update( **train_params )

verbose = False
if debugging:
    verbose = True

trainer   = eb.eb_trainer(parameter, machine, ds_train,
                          ds_valid = None,
                          backup_location = savedir,
                          backup_interval = 20000,
                          hess_interval   = compute_hessian,
                          verbose = verbose)

print "============================================================"
print "EXPERIMENT: #%d (%s)" % (exper_nr, savedir or 'manual run')
print "============================================================"

trainer.train(train_iters)

if savedir is not None:
    trainer.backup_machine(savedir + '/machine.obj')
    trainer.backup_stats(savedir + '/train_stats.obj')

######################################################################
# TESTING

print "============================================================"
print "TESTING"
print "============================================================"

input, output, target = map(eb.state, (shape_in, shape_out, shape_out))

num_classes = len(all_classes)
confusion   = np.zeros((num_classes, num_classes), 'i')

correct = 0
for i in ds_test.iterall(input, target):
    classifier.fprop(input, output)
    ct = target.x.argmax()
    co = output.x.argmax()
    confusion[ct, co] += 1
    if ct == co: correct += 1

print 'Accuracy = %.1f%%' % ((100.0 * correct) / len(ds_test))
print 'Confusion Matrix:'
for crow in confusion:
    occurs = np.sum(crow)
    line = ' '.join(['%-10.3g' % (float(c)/occurs,) for c in crow])
    print line + '    (occurred %d times)' % (occurs,)

