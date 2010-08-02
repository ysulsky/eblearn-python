from params import *

import eblearn    as eb
import eblearn.ui as ui
import numpy      as np

######################################################################
# DATASOURCE

train_mat  = eb.map_matrix('%s/%s' % (data_dir, train_file))
train_ds   = eb.dsource_unsup(train_mat)

if bias is None or coeff is None:
    train_ds.normalize(scalar_bias = True, scalar_coeff = True)
    
if bias  is not None: train_ds.bias  = bias
if coeff is not None: train_ds.coeff = coeff

shape_in, shape_out = train_ds.shape()

######################################################################
# ENCODER

conv_kernel     = (ki, kj)
conv_conn_table = eb.convolution.full_table(shape_in[0], size_code)
conv_out_size   = (size_code,)+tuple(1+np.subtract(shape_in[1:], conv_kernel))

encoder = None
if encoder_arch == 0:
    # LINEAR + BIAS + TANH + DIAG
    encoder = eb.layers(eb.convolution  (conv_kernel, conv_conn_table),
                        eb.bias_module  (conv_out_size, per_feature = True),
                        eb.transfer_tanh(),
                        eb.diagonal     (conv_out_size))
else:
    raise NotImplementedError("this encoder setup isn't implemented yet")

######################################################################
# DECODER

back_conv_conn_table = eb.back_convolution.decoder_table(conv_conn_table)
decoder = eb.back_convolution(conv_kernel, back_conv_conn_table)

######################################################################
# COSTS

# encoder cost
enc_cost = eb.distance_l2(average=False)

# reconstruction cost
bconv_rec_coeff = eb.bconv_rec_cost.coeff_from_conv(shape_out, (1,)+conv_kernel)
rec_cost = eb.bconv_rec_cost(bconv_rec_coeff, average=False)

# sparsity penalty
sparsity = None
if pool_cost == 0:
    sparsity = eb.penalty_l1(0., average=False)
else:
    raise NotImplementedError("this code pooling isn't implemented yet")

######################################################################
# TRAINING

machine = eb.psd_codec( encoder, enc_cost,
                        sparsity,
                        decoder, rec_cost,
                        alphae, alphaz, alphad )

if encoder_arch in (1, 2, 4, 6):
    # ensure a positive code for these machines
    machine.code_parameter.thresh_x = 0.

machine.code_parameter.updater = eb.gd_linesearch_update( machine.code_feval,
                                                          **minimize_code     )
machine.code_trainer.keep_log = True

encoder.parameter.updater = eb.gd_update( **train_encoder )
decoder.parameter.updater = eb.gd_update( **train_decoder )

all_params = eb.parameter_container( encoder.parameter, decoder.parameter )
trainer = eb.eb_trainer(all_params, machine, train_ds,
                        do_normalization = True,
                        backup_location = savedir,
                        backup_interval = 20000,
                        hess_interval   = compute_hessian,
                        verbose = True)

encoder.parameter.name = 'encoder-param'
decoder.parameter.name = 'decoder-param'

print "============================================================"
print "EXPERIMENT: #%d (%s)" % (exper_nr, savedir or 'manual run')
print "============================================================"
trainer.train(train_iters)

if savedir is not None:
    eb.save_object(machine, savedir + '/machine.obj')


def plot():
    import plots
    ui.new_window()
    scale = 3
    plots.plot_filters(machine.encoder, conv_kernel, scale=scale)
    plots.plot_filters(machine.decoder, conv_kernel,
                       transpose = True, orig_x=405, scale=scale)

    ui.new_window()
    plots.plot_reconstructions(train_ds, machine, n=100, scale=scale)


if __name__ == '__main__':
    plot()
    raw_input('press return to exit> ')
