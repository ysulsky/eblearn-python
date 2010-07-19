from eblearn import *

def map_inputs(f, x):
    return array([f(i) for i in x])

def func1(input):
    weights = array(xrange(1,input.size+1))
    return sp.dot(weights, input.ravel())


func = func1

train_data = sp.random.random((100,10))
valid_data = sp.random.random((50,10))

ds_train = dsource_sup(train_data, map_inputs(func, train_data))
ds_train.normalize()

ds_valid = dsource_sup(valid_data, map_inputs(func, valid_data),
                       bias = ds_train.bias, coeff = ds_train.coeff)

shape_in, shape_out = ds_train.shape()
machine = layers(linear(shape_in, shape_out), bias(shape_out))
cost    = distance_l2()

upd = parameter_update(eta = 0.001)

trainer = eb_trainer(ebm_2(machine, cost), upd, ds_train, 
                     ds_valid = ds_valid,
                     backup_location = '/tmp',
                     backup_interval = 2000,
#                    hess_interval = 0,
)

trainer.train(100000)
