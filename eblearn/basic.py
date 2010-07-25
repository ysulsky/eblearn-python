from module import *
from correlate import correlate_table_for_dim, back_correlate_table_for_dim

class linear (module_1_1):
    def __init__(self, shape_in, shape_out):
        ''' out[] = w[][] . in[] '''
        self.shape_in  = ensure_tuple(shape_in)
        self.shape_out = ensure_tuple(shape_out)
        size_in  = product(shape_in)
        size_out = product(shape_out)
        self.w = self.param((size_out, size_in))

    def forget(self):
        arg = self.forget_param
        fanin = self.w.shape[1]
        z = arg.lin_value / (fanin ** (1.0 / arg.lin_exponent))
        self.w.x = sp.random.random(self.w.shape) * (2*z) - z

    def normalize(self):
        for col in self.w.x.T:
            col *= 1.0 / sqrt(sqmag(col))

    def fprop(self, input, output):
        assert (self.shape_in == input.shape)
        output.resize(self.shape_out)
        m2dotm1(self.w.x, input.x.ravel(), output.x.ravel())

    def bprop_input(self, input, output):
        m2dotm1(self.w.x.T, output.dx.ravel(), input.dx.ravel(), True)
    def bprop_param(self, input, output):
        self.w.dx += sp.outer(output.dx.ravel(), input.x.ravel())

    def bbprop_input(self, input, output):
        iddx, oddx = input.ddx.ravel(), output.ddx.ravel()
        m2dotm1(sp.square(self.w.x.T), oddx, iddx, True)
    def bbprop_param(self, input, output):
        self.w.ddx += sp.outer(output.ddx.ravel(), sp.square(input.x.ravel()))


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
        self.b.x = sp.random.random(self.b.shape) * (2*z) - z
    
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
        sp.multiply(input.x, self.d.x, output.x)
    
    def bprop_input(self, input, output):
        input.dx += output.dx * self.d.x
    def bprop_param(self, input, output):
        ix  = input.x.reshape  ((len(self.d), -1))
        odx = output.dx.reshape((len(self.d), -1))
        rdx = self.d.dx.ravel()
        rdx += (ix * odx).sum(1)
    
    def bbprop_input(self, input, output):
        input.ddx += output.ddx * sp.square(self.d.x)
    def bbprop_param(self, input, output):
        ix   = input.x.reshape   ((len(self.d), -1))
        oddx = output.ddx.reshape((len(self.d), -1))
        rddx = self.d.ddx.ravel()
        rddx += (sp.square(ix) * oddx).sum(1)

class convolution (module_1_1):
    @staticmethod
    def full_table(feat_in, feat_out):
        return [(a,b)
                for a in xrange(feat_in)
                for b in xrange(feat_out)]
    
    @staticmethod
    def rand_table(feat_in, feat_out, fanin):
        tab        = sp.empty((fanin * feat_out,2), dtype=int)
        out_col    = unfold(tab[:,1], 0, fanin,   fanin)
        out_col[:] = sp.arange(feat_out).reshape(feat_out,1)
        in_col     = unfold(tab[:,0], 0, feat_in, feat_in)
        for chunk in in_col:
            chunk[:] = sp.random.permutation(feat_in)
        if len(tab) > len(in_col):
            remainder = tab[in_col.size:,0]
            remainder[:] = sp.random.permutation(feat_in)[:len(remainder)]
        
        tab = map(tuple, tab)
        tab.sort()
        return tab
    
    def __init__(self, kernel_shape, conn_table):
        ''' out[j][] += in[i][] <*> kernel[k][]
            where conn_table[k] = (i,j) '''
        self.conn_table = sp.asarray(conn_table, int)
        self.kernels    = self.param((len(conn_table),) + kernel_shape)
        self.feat_out   = 1 + self.conn_table[:,1].max()
        self.fanin      = sp.zeros(self.feat_out, dtype=int)
        for j in self.conn_table[:,1]: self.fanin[j] += 1
        
        ndim = len(kernel_shape)
        self.fcorr   = correlate_table_for_dim(ndim)
        self.bcorr   = back_correlate_table_for_dim(ndim)

        e = enumerate
        self.tbl_ikj = sp.asarray([(a,k,b) for (k,(a,b)) in e(conn_table)], 'i')
        self.tbl_jki = sp.asarray([(b,k,a) for (k,(a,b)) in e(conn_table)], 'i')
        self.tbl_ijk = sp.asarray([(a,b,k) for (k,(a,b)) in e(conn_table)], 'i')
    
    def forget(self):
        arg = self.forget_param
        p = 1. / arg.lin_exponent
        for j, kx in zip( self.conn_table[:,1], self.kernels.x ):
            z = arg.lin_value / ((self.fanin[j] * kx.size) ** p)
            kx[:] = sp.random.random(kx.shape) * (2*z) - z

    def normalize(self):
        for kx in self.kernels.x: kx /= sqrt(sqmag(kx))

    def fprop(self, input, output):
        out_shape = sp.subtract(input.shape[1:], self.kernels.shape[1:]) + 1
        output.resize((self.feat_out,) + tuple(out_shape))
        clear(output.x)
        self.fcorr(self.tbl_ikj, input.x, self.kernels.x, output.x)
            
    def bprop_input(self, input, output):
        self.bcorr(self.tbl_jki, output.dx, self.kernels.x, input.dx)
    def bprop_param(self, input, output):
        self.fcorr(self.tbl_ijk, input.x, output.dx, self.kernels.dx)
    
    def bbprop_input(self, input, output):
        sq = sp.square
        self.bcorr(self.tbl_jki, output.ddx, sq(self.kernels.x), input.ddx)
    def bbprop_param(self, input, output):
        sq = sp.square
        self.fcorr(self.tbl_ijk, sq(input.x), output.ddx, self.kernels.ddx)


class back_convolution (convolution):
    @staticmethod
    def decoder_table(encoder_table):
        return [(b,a) for (a,b) in encoder_table]

    def __init__(self, kernel_shape, conn_table):
        super(back_convolution, self).__init__(kernel_shape, conn_table)

        e = enumerate
        self.tbl_jik = sp.asarray([(b,a,k) for (k,(a,b)) in e(conn_table)], int)
        
    def fprop(self, input, output):
        out_shape = sp.subtract(input.shape[1:], 1) + self.kernels.shape[1:]
        output.resize((self.feat_out,) + tuple(out_shape))
        clear(output.x)
        self.bcorr(self.tbl_ikj, input.x, self.kernels.x, output.x)
    
    def bprop_input(self, input, output):
        self.fcorr(self.tbl_jki, output.dx, self.kernels.x, input.dx)
    def bprop_param(self, input, output):
        self.fcorr(self.tbl_jik, output.dx, input.x, self.kernels.dx)

    def bbprop_input(self, input, output):
        sq = sp.square
        self.fcorr(self.tbl_jki, output.ddx, sq(self.kernels.x), input.ddx)
    def bbprop_param(self, input, output):
        sq = sp.square
        self.fcorr(self.tbl_jik, output.ddx, sq(input.x), self.kernels.ddx)


class multiplication (no_params, module_2_1):
    def fprop(self, input1, input2, output):
        assert(input1.shape == input2.shape)
        output.resize(input1.shape)
        sp.multiply(input1.x, input2.x, output.x)
    def bprop_input(self, input1, input2, output):
        input1.dx += output.dx * input2.x
        input2.dx += output.dx * input1.x
    def bbprop_input(self, input1, input2, output):
        input1.ddx += output.ddx * sp.square(input2.x)
        input2.ddx += output.ddx * sp.square(input1.x)
        
        
