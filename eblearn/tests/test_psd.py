#!/usr/bin/env python

from eblearn     import *

def gen_sample():
    x = zeros((16,16))
    row = x[np.random.randint(len(x))]
    row.fill(1.)
    return x

def plot_filters(m, shape_in,
                 transpose=False, orig_x=5, orig_y=5, max_x=390, scale = 1.0):
    assert (len(shape_in) == 2)
    ensure_window(title = 'PSD Filters')
    h, w = shape_in; h *= scale; w *= scale
    padding, pos_x, pos_y = 5, 0, 0
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

def plot_reconstructions(ds, machine, n = 1000,
                         orig_x=5, orig_y=5, max_x=795, scale = 1.0):
    shape_in, shape_out = ds.shape()
    assert (len(shape_in) == len(shape_out) == 2)
    ensure_window(title = 'PSD Reconstructions')
    spacing, padding = 2, 5
    pic = empty((max(shape_in[0], shape_out[0]),
                 shape_in[1] + spacing + shape_out[1]))
    h, w = pic.shape; h *= scale; w *= scale
    pos_x, pos_y = 0, 0
    ds.seek(0)
    inp, tgt = state(()), state(())
    for i in xrange(min(n, ds.size())):
        ds.fprop(inp, tgt)
        machine.encoder.fprop(inp, machine.encoder_out)
        machine.decoder.fprop(machine.encoder_out, machine.decoder_out)
        ds.next()
        inpx = inp.x
        recx = machine.decoder_out.x
        white = max( inpx.max(), recx.max() )
        pic.fill(white)
        pic[:shape_in[0],:shape_in[1]] = inpx
        pic[:shape_out[0],shape_in[1] + spacing:] = recx
        if pos_x + w > max_x:
            pos_x  = 0; pos_y += h + padding
        draw_mat(pic, orig_x + pos_x, orig_y + pos_y,
                 maxv = white, scale = scale)
        pos_x += w + padding
        

hidden    = 0
code_size = 256

ds_train = dsource_unsup(array([gen_sample() for i in xrange(200)]))
ds_train.normalize(scalar_bias = True, scalar_coeff = True)

ds_valid = None

shape_in, shape_out = ds_train.shape()

encoder = None
if hidden > 0:
    encoder = layers( linear        (shape_in, hidden),
                      bias_module   (hidden),
                      transfer_tanh (),
                      linear        (hidden, code_size),
                      bias_module   (code_size),
                      transfer_tanh (),
                      diagonal      (code_size)          )
else:
    encoder = layers( linear        (shape_in, code_size),
                      bias_module   (code_size),
                      transfer_tanh (),
                      diagonal      (code_size)          )

decoder = layers( linear      (code_size, shape_out),
                  bias_module (shape_out)             )


machine = psd_codec( encoder, distance_l2(), penalty_l1(),
                     decoder, distance_l2() )

machine.code_parameter.updater = gd_linesearch_update( machine.code_feval,
                                                       eta = 1.0,
                                                       #norm_grad = True,
                                                       max_line_steps = 10,
                                                       max_iters   = 100,
                                                       grad_thresh = 0.001,
                                                       anneal_time = 10,
                                                       anneal_amt  = 0.1 ,
                                                       #quiet = False
                                                       )
encoder.parameter.updater = gd_update( eta = 0.1 )
decoder.parameter.updater = gd_update( eta = 0.01 )

trainer = eb_trainer(parameter_container(encoder.parameter, decoder.parameter),
                     machine, ds_train, 
                     do_normalization = True,
                     ds_valid = ds_valid,
                     backup_location = '/tmp',
#                    backup_interval = 2000,
                     hess_interval = 10000,
                     report_interval = 50,
)

trainer.train(2000)

new_window()
plot_filters(machine.encoder, shape_in)
plot_filters(machine.decoder, shape_out, transpose = True, orig_x=405)

new_window()
plot_reconstructions(ds_train, machine, 1000)

