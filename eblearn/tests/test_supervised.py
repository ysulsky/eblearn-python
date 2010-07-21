from eblearn import *

def map_inputs(f, x):
    return array([f(i) for i in x])

def func(x):
    return 10 * sp.sin(x * (2*pi))

def plot(ds, machine):
    shape_in, shape_out = ds.shape()
    assert (shape_in == shape_out == (1,))
    inp = state((1,)); out = state((1,)); des = state((1,))
    ds.seek(0)
    size = ds.size()
    coords  = empty(size); outputs = empty(size); targets = empty(size)
    for i in xrange(size):
        ds.fprop(inp, des)
        machine.fprop(inp, out)
        coords[i] = inp.x; outputs[i] = out.x; targets[i] = des.x
        ds.next()
    from matplotlib import pyplot
    indices = coords.argsort()
    coords  = coords.take(indices)
    outputs = outputs.take(indices)
    targets = targets.take(indices)
    pyplot.ioff()
    pyplot.plot(coords, outputs, label = 'machine output')
    pyplot.plot(coords, targets, label = 'desired output')
    pyplot.legend()
    pyplot.show()

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

param = machine.parameter
param.updater = gd_update( eta = 0.02 )

trainer = eb_trainer(param, ebm_2(machine, cost), ds_train, 
                     ds_valid = ds_valid,
                     backup_location = '/tmp',
#                    backup_interval = 2000,
#                    hess_interval = 0,
#                    report_interval = 1
)

trainer.train(10000)

plot(ds_valid, machine)
