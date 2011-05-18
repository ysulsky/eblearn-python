from eblearn.correlate import config_back_convolve_table,  \
                              config_convolve_table,       \
                              config_back_correlate_table, \
                              config_correlate_table
from eblearn.idx       import reverse, reverse_along, unfold
from eblearn.module    import module_1_1, module_2_1, no_params
from eblearn.util      import ensure_tuple, random
from eblearn.vecmath   import clear, m2dotm1, m2dotrows, normrows

import numpy as np

class linear (module_1_1):
    def __init__(self, shape_in, shape_out):
        ''' out[] = w[][] . in[] '''
        self.shape_in  = ensure_tuple(shape_in)
        self.shape_out = ensure_tuple(shape_out)
        size_in  = np.prod(shape_in)
        size_out = np.prod(shape_out)
        self.w = self.param((size_out, size_in))

    def forget(self):
        arg = self.forget_param
        fanin = self.w.shape[1]
        z = arg.lin_value / (fanin ** (1.0 / arg.lin_exponent))
        self.w.x = random(self.w.shape) * (2*z) - z

    def normalize(self):
        normrows(self.w.x.T)
    
    def fprop(self, input, output):
        assert (self.shape_in == input.shape)
        output.resize(self.shape_out)
        m2dotm1(self.w.x, input.x.ravel(), output.x.ravel())
    
    def bprop_input(self, input, output):
        m2dotm1(self.w.x.T, output.dx.ravel(), input.dx.ravel(), True)
    def bprop_param(self, input, output):
        self.w.dx += np.outer(output.dx.ravel(), input.x.ravel())
    
    def bbprop_input(self, input, output):
        iddx, oddx = input.ddx.ravel(), output.ddx.ravel()
        m2dotm1(np.square(self.w.x.T), oddx, iddx, True)
    def bbprop_param(self, input, output):
        self.w.ddx += np.outer(output.ddx.ravel(), np.square(input.x.ravel()))


class bias_module (module_1_1):
    def __init__(self, shape_in, per_feature = False):
        ''' if per_feature is false, out[]    = in[]    + b[]
            otherwise                out[k][] = in[k][] + b[k] '''

        self.per_feature = per_feature
        self.shape_in    = ensure_tuple(shape_in)
        shape_b = self.shape_in
        if per_feature:
            shape_b = (shape_b[0],) + (len(shape_b)-1)*(1,)
        self.b  = self.param(shape_b)
    
    def forget(self):
        arg = self.forget_param
        z = arg.lin_value
        self.b.x = random(self.b.shape) * (2*z) - z
    
    def normalize(self): pass
    
    def fprop(self, input, output):
        assert (self.shape_in == input.shape)
        output.resize(input.shape)
        output.x[:] = input.x + self.b.x
    
    def bprop_input(self, input, output):
        input.dx += output.dx
    def bprop_param(self, input, output):
        odx = output.dx
        if self.per_feature:
            odx = odx.reshape((len(self.b), -1)).sum(1).reshape(self.b.shape)
        self.b.dx += odx
    
    def bbprop_input(self, input, output):
        input.ddx  += output.ddx
    def bbprop_param(self, input, output):
        oddx = output.ddx
        if self.per_feature:
            oddx = oddx.reshape((len(self.b), -1)).sum(1).reshape(self.b.shape)
        self.b.ddx += oddx

class diagonal (module_1_1):
    def __init__ (self, shape_in):
        ''' out[k][] = in[k][] * d[k] '''
        self.shape_in = ensure_tuple(shape_in)
        shape_d = (self.shape_in[0],) + (len(self.shape_in)-1)*(1,)
        self.d  = self.param(shape_d)

    def forget(self):
        self.d.x.fill(1.)
    
    def normalize(self): pass

    def fprop(self, input, output):
        assert (self.shape_in == input.shape)
        output.resize(input.shape)
        np.multiply(input.x, self.d.x, output.x)
    
    def bprop_input(self, input, output):
        input.dx += output.dx * self.d.x
    def bprop_param(self, input, output):
        ix  = input.x.reshape  ((len(self.d), -1))
        odx = output.dx.reshape((len(self.d), -1))
        m2dotrows(ix, odx, self.d.dx.ravel(), True)
    
    def bbprop_input(self, input, output):
        input.ddx += output.ddx * np.square(self.d.x)
    def bbprop_param(self, input, output):
        ix   = input.x.reshape   ((len(self.d), -1))
        oddx = output.ddx.reshape((len(self.d), -1))
        m2dotrows(np.square(ix), oddx, self.d.ddx.ravel(), True)

class convolution (module_1_1):
    @staticmethod
    def full_table(feat_in, feat_out):
        return [(a,b)
                for a in xrange(feat_in)
                for b in xrange(feat_out)]
    
    @staticmethod
    def rand_table(feat_in, feat_out, fanin):
        tab        = np.empty((fanin * feat_out,2), dtype=int)
        out_col    = unfold(tab[:,1], 0, fanin,   fanin)
        out_col[:] = np.arange(feat_out).reshape(feat_out,1)
        in_col     = unfold(tab[:,0], 0, feat_in, feat_in)
        for chunk in in_col:
            chunk[:] = np.random.permutation(feat_in)
        if len(tab) > len(in_col):
            remainder = tab[in_col.size:,0]
            remainder[:] = np.random.permutation(feat_in)[:len(remainder)]
        
        tab = map(tuple, tab)
        tab.sort()
        return tab
    
    def __init__(self, kernel_shape, conn_table, correlation=False):
        ''' out[j][] += in[i][] <*> kernel[k][]
            where conn_table[k] = (i,j)
             
            correlation=True -> <*> is correlation instead of convolution
        '''
        kernel_shape = ensure_tuple(kernel_shape)
        
        self.conn_table = np.asarray(conn_table, int)
        self.correlation= correlation
        self.kernels    = self.param((len(conn_table),) + kernel_shape)
        self.feat_out   = 1 + self.conn_table[:,1].max()
        self.fanin      = np.zeros(self.feat_out, dtype=int)
        for j in self.conn_table[:,1]: self.fanin[j] += 1
        
        # we can configure these further, but then we would need to use
        # individual functions per table
        ndim = len(kernel_shape)
        if correlation:
            self.fconv   = config_correlate_table(ndim)
            self.bconv   = config_back_correlate_table(ndim)
        else:
            self.fconv   = config_convolve_table(ndim)
            self.bconv   = config_back_convolve_table(ndim)
        
        e = enumerate
        self.tbl_ikj = np.asarray([(a,k,b) for (k,(a,b)) in e(conn_table)], 'i')
        self.tbl_jki = np.asarray([(b,k,a) for (k,(a,b)) in e(conn_table)], 'i')
        self.tbl_ijk = np.asarray([(a,b,k) for (k,(a,b)) in e(conn_table)], 'i')
        
        self.dsize = 1 - np.asarray((1,) + kernel_shape, 'i')
    
    def forget(self):
        arg = self.forget_param
        p = 1. / arg.lin_exponent
        for j, kx in zip( self.conn_table[:,1], self.kernels.x ):
            z = arg.lin_value / ((self.fanin[j] * kx.size) ** p)
            kx[:] = random(kx.shape) * (2*z) - z

    def normalize(self):
        normrows(self.kernels.x)
    
    def fprop(self, input, output):
        out_shape    = self.dsize + input.shape
        out_shape[0] = self.feat_out
        output.resize(out_shape)
        clear(output.x)
        self.fconv(self.tbl_ikj, input.x, self.kernels.x, output.x)
            
    def bprop_input(self, input, output):
        self.bconv(self.tbl_jki, output.dx, self.kernels.x, input.dx)
    def bprop_param(self, input, output):
        if self.correlation:
            self.fconv(self.tbl_ijk, input.x, output.dx, self.kernels.dx)
        else:
            rev_odx = reverse_along(reverse(output.dx), 0)
            rev_kdx = reverse_along(reverse(self.kernels.dx), 0)
            self.fconv(self.tbl_ijk, input.x, rev_odx, rev_kdx)
    
    def bbprop_input(self, input, output):
        sq = np.square
        self.bconv(self.tbl_jki, output.ddx, sq(self.kernels.x), input.ddx)
    def bbprop_param(self, input, output):
        sq = np.square
        if self.correlation:
            self.fconv(self.tbl_ijk, sq(input.x), output.ddx, self.kernels.ddx)
        else:
            rev_oddx = reverse_along(reverse(output.ddx), 0)
            rev_kddx = reverse_along(reverse(self.kernels.ddx), 0)
            self.fconv(self.tbl_ijk, sq(input.x), rev_oddx, rev_kddx)



class back_convolution (convolution):
    @staticmethod
    def decoder_table(encoder_table):
        return [(b,a) for (a,b) in encoder_table]

    def __init__(self, kernel_shape, conn_table, correlation=False):
        kernel_shape = ensure_tuple(kernel_shape)
        super(back_convolution, self).__init__(kernel_shape,
                                               conn_table, correlation)
        e = enumerate
        self.tbl_jik = np.asarray([(b,a,k) for (k,(a,b)) in e(conn_table)], 'i')
        self.dsize = np.asarray((1,) + kernel_shape, 'i') - 1
        
    def fprop(self, input, output):
        out_shape    = self.dsize + input.shape
        out_shape[0] = self.feat_out
        output.resize(out_shape)
        clear(output.x)
        self.bconv(self.tbl_ikj, input.x, self.kernels.x, output.x)
    
    def bprop_input(self, input, output):
        self.fconv(self.tbl_jki, output.dx, self.kernels.x, input.dx)
    def bprop_param(self, input, output):
        if self.correlation:
            self.fconv(self.tbl_jik, output.dx, input.x, self.kernels.dx)
        else:
            rev_ix  = reverse_along(reverse(input.x), 0)
            rev_kdx = reverse_along(reverse(self.kernels.dx), 0)
            self.fconv(self.tbl_jik, output.dx, rev_ix, rev_kdx)

    def bbprop_input(self, input, output):
        sq = np.square
        self.fconv(self.tbl_jki, output.ddx, sq(self.kernels.x), input.ddx)
    def bbprop_param(self, input, output):
        sq = np.square
        if self.correlation:
            self.fconv(self.tbl_jik, output.ddx, sq(input.x), self.kernels.ddx)
        else:
            rev_sqix  = reverse_along(reverse(sq(input.x)), 0)
            rev_kddx = reverse_along(reverse(self.kernels.ddx), 0)
            self.fconv(self.tbl_jik, output.ddx, rev_sqix, rev_kddx)


class multiplication (no_params, module_2_1):
    def fprop(self, input1, input2, output):
        assert(input1.shape == input2.shape)
        output.resize(input1.shape)
        np.multiply(input1.x, input2.x, output.x)
    def bprop_input(self, input1, input2, output):
        input1.dx += output.dx * input2.x
        input2.dx += output.dx * input1.x
    def bbprop_input(self, input1, input2, output):
        input1.ddx += output.ddx * np.square(input2.x)
        input2.ddx += output.ddx * np.square(input1.x)
        
        
class concatenation (no_params, module_2_1):
    def fprop(self, input1, input2, output):
        assert(input1.shape[1:] == input2.shape[1:])
        output.resize((input1.shape[0] + input2.shape[0],) + input1.shape[1:])
        output.x[:] = np.concatenate((input1.x, input2.x))
    def bprop_input(self, input1, input2, output):
        n = len(input1.dx)
        input1.dx += output.dx[:n]
        input2.dx += output.dx[n:]
    def bbprop_input(self, input1, input2, output):
        n = len(input1.ddx)
        input1.ddx += output.ddx[:n]
        input2.ddx += output.ddx[n:]
