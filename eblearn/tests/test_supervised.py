from eblearn import *

def map_inputs(f, x):
    return array([f(i) for i in x])

def func(x):
    return 10 * sp.sin(x * (2*pi))

hidden = (512,)

train_data = sp.random.random((100,1))
valid_data = sp.random.random((50,1))

ds_train = dsource_sup(train_data, map_inputs(func, train_data))
ds_train.normalize()

ds_valid = dsource_sup(valid_data, map_inputs(func, valid_data),
                       bias = ds_train.bias, coeff = ds_train.coeff)

shape_in, shape_out = ds_train.shape()
machine = layers( linear(shape_in, hidden),
                  bias(hidden),
                  transfer_tanh(),
                  linear(hidden, shape_out),
                  bias(shape_out) )

cost    = distance_l2()

upd = parameter_update(eta = 0.05)

trainer = eb_trainer(ebm_2(machine, cost), upd, ds_train, 
                     ds_valid = ds_valid,
                     backup_location = '/tmp',
#                    backup_interval = 2000,
#                    hess_interval = 0,
#                    report_interval = 1
)

trainer.train(50000)
