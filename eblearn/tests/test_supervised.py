from eblearn      import *
from eblearn.util import *

def map_inputs(f, x):
    return array([f(i) for i in x])

def func_sq(x):
    return np.square((x - .5) * 10)

def func_sin(x):
    return 10 * np.sin(x * (2*pi))

func = func_sin

def plot(machine, train_ds, valid_ds = None):
    from matplotlib import pyplot
    
    shape_in, shape_out = train_ds.shape()
    assert (shape_in == shape_out == (1,))
    inp = state((1,)); out = state((1,)); des = state((1,))

    all_ds = [train_ds]+([valid_ds] if valid_ds else [])
    fullsize = sum([ds.size() for ds in all_ds])
    coords  = empty(fullsize)
    outputs = empty(fullsize)
    targets = empty(fullsize)
    offset = 0
    for ds in all_ds:
        ds.seek(0)
        size = ds.size()
        for i in xrange(offset, offset+size):
            ds.fprop(inp, des)
            machine.fprop(inp, out)
            coords[i] = inp.x; outputs[i] = out.x; targets[i] = des.x
            ds.next()
        offset += size

    pyplot.ioff()
    
    offset = 0
    if train_ds:
        size = train_ds.size()
        ds_coords, ds_targets = \
            narrow(coords, 0, size, offset), narrow(targets, 0, size, offset)
        indices = ds_coords.argsort()
        pyplot.plot(ds_coords.take(indices), ds_targets.take(indices),
                    'rx', label = 'training points')
        offset += size

    if valid_ds:
        size = valid_ds.size()
        ds_coords, ds_targets = \
            narrow(coords, 0, size, offset), narrow(targets, 0, size, offset)
        indices = ds_coords.argsort()
        pyplot.plot(ds_coords.take(indices), ds_targets.take(indices),
                    'gx', label = 'validation points')
        offset += size

    indices = coords.argsort()
    coords  = coords.take(indices)
    outputs = outputs.take(indices)
    pyplot.plot(coords, outputs, 'b',  label = 'machine output')
    pyplot.legend()
    pyplot.show()

linesearch = False
hidden = 512

train_data = random((100,1))
valid_data = random((50,1))

ds_train = dsource_sup(train_data, map_inputs(func, train_data))
ds_train.normalize()

ds_valid = dsource_sup(valid_data, map_inputs(func, valid_data),
                       bias = ds_train.bias, coeff = ds_train.coeff)

shape_in, shape_out = ds_train.shape()
machine = layers( linear(shape_in, hidden),
                  bias_module(hidden),
                  transfer_tanh(),
                  linear(hidden, shape_out),
                  bias_module(shape_out)    )

cost    = distance_l2()

param = machine.parameter

hessian_interval = 2000 # 0 disables

trainer = eb_trainer(param, ebm_2(machine, cost, name='machine+cost'), ds_train,
                     ds_valid = ds_valid,
                     backup_location = '/tmp',
                     backup_interval = 2000,
                     hess_interval = hessian_interval,
                     verbose = True,
#                    report_interval = 1,
)

gd_params = dict (
    eta = 0.5 if linesearch else 0.01
,   debugging = True
)

if linesearch:
    feval = feval_from_trainer(trainer)
    #gd_params['quiet']  = False
    param.updater = gd_linesearch_update(feval, **gd_params)
else:
    param.updater = gd_update(**gd_params)


trainer.train(10000)

plot(machine, ds_train, ds_valid)
