from eblearn     import *
from eblearn.psd import *

def gen_sample():
    x = zeros((16,16))
    row = x[sp.random.randint(len(x))]
    row.fill(1.)
    return x

def plot_filters(m, shape_in,
                 transpose=False, orig_x=0, orig_y=5, max_x=400, scale = 1.0):
    assert (len(shape_in) == 2)
    h, w = shape_in; h *= scale; w *= scale
    padding = 5
    pos_x, pos_y = 0, 0
    filters =m.parameter.states[0].x
    if transpose: filters = filters.T
    minv = filters.min()
    maxv = filters.max()
    for k in filters:
        k = k.reshape(shape_in)
        if pos_x + w > max_x:
            pos_x  = 0; pos_y += h + padding
        draw_mat(k, orig_x + pos_x, orig_y + pos_y,
                 minv = minv, maxv = maxv, scale = scale)
        pos_x += w + padding
    pos_y += h + padding + 16
    draw_text(orig_x, orig_y + pos_y, 'Values range in %g .. %g' % (minv,maxv))


hidden    = 0
code_size = 256

ds_train = dsource_unsup(array([gen_sample() for i in xrange(200)]))
ds_train.normalize(scalar_bias = True, scalar_coeff = True)

ds_valid = None

shape_in, shape_out = ds_train.shape()

encoder = None
if hidden > 0:
    encoder = layers( linear        (shape_in, hidden),
                      bias          (hidden),
                      transfer_tanh (),
                      linear        (hidden, code_size),
                      bias          (code_size)          )
else:
    encoder = layers( linear        (shape_in, code_size),
                      bias          (code_size),
                      transfer_tanh ()                   )

# TODO: add diag-module here, for both encoders                      

decoder = layers( linear (code_size, shape_out),
                  bias   (shape_out)             )


encoder.parameter.indep = True
decoder.parameter.indep = True

machine = psd_codec( encoder, distance_l2(), penalty_l1(),
                     decoder, distance_l2() )

# TODO: separately train encoder and decoder
upd     = parameter_update( eta = 0.05 )
trainer = eb_trainer(machine, upd, ds_train, 
                     ds_valid = ds_valid,
                     backup_location = '/tmp',
#                    backup_interval = 2000,
#                    hess_interval = 0,
                     report_interval = 100
)

trainer.train(5000)
plot_filters(machine.encoder, shape_in)
plot_filters(machine.decoder, shape_out, transpose = True, orig_x=400)
