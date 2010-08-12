#!./run ##############################################################

exper_nr  = 1

debugging    = False

######################################################################
# CLASSIFIER

encoder_arch    = 1        # 1 -> linear
                           # 2 -> 2-layer mlp

cost_type       = 1        # 1 -> cross-entropy
                           # 2 -> euclidean distance

######################################################################
# TRAINING

train_iters     = 50000    # number of training iterations
compute_hessian = 5000     # hessian update interval (disable with 0)

train_params = dict (
    eta         = 0.02
,   anneal_amt  = 0.1
,   anneal_time = 1000
,   decay_l2    = 0.0      # L2 regularization
,   decay_l1    = 0.0      # L1 regularization
,   decay_time  = 1000
,   debugging = debugging
)

######################################################################
# DATA

data_file       = './iris.data'

######################################################################
script   = './run_classifier.py'
outputs  = './outputs'

savedir  = None                                 # set by the run script
exper_id = 'exper%d (manual run)' % (exper_nr,) # ditto

