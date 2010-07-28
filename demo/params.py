exper  = 1

debugging    = False
eblearn_path = '../'
import sys; sys.path.append(eblearn_path)

######################################################################
# MACHINE

ki              = 3        # number of rows in the convolution kernel
kj              = 3        # number of cols in the convolution kernel
size_code       = 256      # size of hidden units
alphae          = 1        # encoder energy contribution
alphad          = 1        # reconstruction energy contribution
alphaz          = 1        # sparsity penalty energy contribution

encoder_arch    = 0        # 0 for c-b-th-d
pool_cost       = 0        # 0 for no pooling


######################################################################
# TRAINING

train_iters     = 50000    # over patches
compute_hessian = 5000     # hessian update interval (disable with 0)

train_encoder = dict (
    eta         = 0.02
,   anneal_amt  = 0.1
,   anneal_time = 1000
,   decay_l2    = 0.0      # L2 regularization
,   decay_l1    = 0.0      # L1 regularization
,   decay_time  = 1000
,   debugging = debugging
)

train_decoder = dict (
    eta         = 0.002
,   anneal_amt  = 0.1
,   anneal_time = 1000
,   decay_l2    = 0.0      # L2 regularization
,   decay_l1    = 0.0      # L1 regularization
,   decay_time  = 1000
,   debugging = debugging
)

######################################################################
# CODE MINIMIZATION

minimize_code = dict (
    eta         = 0.5
,   anneal_amt  = 0.1
,   anneal_time = 10
,   max_iters   = 10
,   max_line_steps = 10
,   grad_thresh = 0.001
,   debugging = debugging
)

######################################################################
# DATA

bias            = -110     # datasource bias  (None -> compute from data)
coeff           = 0.0167   # datasource coeff (None -> compute from data)

##### not implemented yet:
# remove-local-mean
# threshold-sample
# sample-std-thres
####

data_dir        = '.'
train_file      = "tr-berkeley-N50K-M9x9.mat"


######################################################################
script  = 'run_psd.py'
outputs = './outputs'

