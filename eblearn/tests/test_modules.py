from eblearn import *

def report_err (a, b, name, tol = 1e-5):
    def report_str_err(err, s):
        print '%-40s = %-15g %15s' % (s, err, "pass" if err < tol else "FAIL")
    max_err = abs(a - b).max()
    tot_err = sqrt(sqdist(a, b))
    report_str_err(max_err, 'Max   %s distance' % name)
    report_str_err(tot_err, 'Total %s distance' % name)

def jacobian_bprop_m_1_1 (mod, sin, sout, jac):
    sout_dx = sout.dx.ravel()
    sin_dx  = sin.dx.ravel()
    for i in xrange(sout.size):
        clear(sin_dx)
        clear(sout_dx)
        sout_dx[i] = 1.
        mod.bprop(sin, sout)
        jac[:,i] = sin_dx

def jacobian_bprop_m_1_1_param (mod, sin, sout, jac):
    sout_dx = sout.dx.ravel()
    for i in xrange(sout.size):
        clear(sout_dx)
        sout_dx[i] = 1.
        mod.parameter.clear_dx()
        mod.bprop(sin, sout)
        jac[:,i] = sp.fromiter(mod.parameter.dx, rtype)

def jacobian_fprop_m_1_1 (mod, sin, sout, jac):
    small = 1e-6
    sina  = state(sin.shape);     souta = state(sout.shape)
    sinb  = state(sin.shape);     soutb = state(sout.shape)
    sin_x = sin.x.ravel();
    sina_x = sina.x.ravel();      sinb_x = sinb.x.ravel()
    sina_x[:] = sin_x;            sinb_x[:] = sin_x
    for i in xrange(sin.size):
        sina_x[i] -= small
        mod.fprop(sina, souta)
        sina_x[i]  = sin_x[i]
        
        sinb_x[i] += small
        mod.fprop(sinb, soutb)
        sinb_x[i]  = sin_x[i]
        
        jac[i,:] = ((soutb.x - souta.x) / (2 * small)).ravel()

def jacobian_fprop_m_1_1_param (mod, sin, sout, jac):
    small = 1e-6
    souta = state(sout.shape)
    soutb = state(sout.shape)
    i = -1
    for pstate in mod.parameter.states:
        for j in xrange(pstate.size):
            i += 1
            pstate_x = pstate.x.ravel()

            pstate_x[j] -= small
            mod.fprop(sin, souta)

            pstate_x[j] += 2 * small
            mod.fprop(sin, soutb)

            pstate_x[j] -= small
            jac[i,:] = ((soutb.x - souta.x) / (2 * small)).ravel()

def test_module_1_1_jac (mod, sin, sout, minval=-2., maxval=2.):
    sin.x = sp.random.random(sin.shape) * (maxval - minval) - minval
    mod.forget()
    mod.fprop(sin, sout)
    insize  = sin.size
    outsize = sout.size
    jac_fprop = zeros((insize, outsize))
    jacobian_fprop_m_1_1(mod, sin, sout, jac_fprop)
    jac_bprop = zeros((insize, outsize))
    jacobian_bprop_m_1_1(mod, sin, sout, jac_bprop)
    report_err(jac_fprop, jac_bprop, "jacobian input")
    return (jac_fprop, jac_bprop)

def test_module_1_1_jac_param (mod, sin, sout, minval=-2., maxval=2.):
    if not mod.has_params(): return
    sin.x = sp.random.random(sin.shape) * (maxval - minval) - minval
    for state in mod.parameter.states:
        state.x = sp.random.random(state.shape) * (maxval - minval) - minval
    mod.fprop(sin, sout)
    insize  = mod.parameter.size()
    outsize = sout.size
    jac_fprop = zeros((insize, outsize))
    jacobian_fprop_m_1_1_param(mod, sin, sout, jac_fprop)
    jac_bprop = zeros((insize, outsize))
    jacobian_bprop_m_1_1_param(mod, sin, sout, jac_bprop)
    report_err(jac_fprop, jac_bprop, "jacobian param")
    return (jac_fprop, jac_bprop)


def jacobian_bprop_m_2_1 (mod, sin1, sin2, sout, jac1, jac2):
    sout_dx = sout.dx.ravel()
    sin1_dx  = sin1.dx.ravel()
    sin2_dx  = sin2.dx.ravel()
    for i in xrange(sout.size):
        clear(sin1_dx)
        clear(sin2_dx)
        clear(sout_dx)
        sout_dx[i] = 1.
        mod.bprop(sin1, sin2, sout)
        jac1[:,i] = sin1_dx
        jac2[:,i] = sin2_dx

def jacobian_bprop_m_2_1_param (mod, sin1, sin2, sout, jac):
    sout_dx = sout.dx.ravel()
    for i in xrange(sout.size):
        clear(sout_dx)
        sout_dx[i] = 1.
        mod.parameter.clear_dx()
        mod.bprop(sin1, sin2, sout)
        jac[:,i] = sp.fromiter(mod.parameter.dx, rtype)

def jacobian_fprop_m_2_1 (mod, sin1, sin2, sout, jac1, jac2):
    small = 1e-6
    sins = [sin1, sin2]; jacs = [jac1, jac2]
    for which in [0, 1]:
        jac = jacs[which]
        sina  = state(sins[which].shape);     souta = state(sout.shape)
        sinb  = state(sins[which].shape);     soutb = state(sout.shape)
        sin_x = sins[which].x.ravel();
        sina_x = sina.x.ravel();              sinb_x = sinb.x.ravel()
        sina_x[:] = sin_x;                    sinb_x[:] = sin_x
        for i in xrange(sin_x.size):
            sina_x[i] -= small
            if which == 0: mod.fprop(sina, sin2, souta)
            else:          mod.fprop(sin1, sina, souta)
            sina_x[i]  = sin_x[i]
        
            sinb_x[i] += small
            if which == 0: mod.fprop(sinb, sin2, soutb)
            else:          mod.fprop(sin1, sinb, soutb)
            sinb_x[i]  = sin_x[i]
        
            jac[i,:] = ((soutb.x - souta.x) / (2 * small)).ravel()


def jacobian_fprop_m_2_1_param (mod, sin1, sin2, sout, jac):
    small = 1e-6
    souta = state(sout.shape)
    soutb = state(sout.shape)
    i = -1
    for pstate in mod.parameter.states:
        for j in xrange(pstate.size):
            i += 1
            pstate_x = pstate.x.ravel()

            pstate_x[j] -= small
            mod.fprop(sin1, sin2, souta)

            pstate_x[j] += 2 * small
            mod.fprop(sin1, sin2, soutb)

            pstate_x[j] -= small
            jac[i,:] = ((soutb.x - souta.x) / (2 * small)).ravel()

def test_module_2_1_jac (mod, sin1, sin2, sout, minval=-2., maxval=2.):
    sin1.x = sp.random.random(sin1.shape) * (maxval - minval) - minval
    sin2.x = sp.random.random(sin2.shape) * (maxval - minval) - minval
    mod.forget()
    mod.fprop(sin1, sin2, sout)
    insize1  = sin1.size
    insize2  = sin2.size
    outsize = sout.size
    jac1_fprop = zeros((insize1, outsize))
    jac2_fprop = zeros((insize2, outsize))
    jacobian_fprop_m_2_1(mod, sin1, sin2, sout, jac1_fprop, jac2_fprop)
    jac1_bprop = zeros((insize1, outsize))
    jac2_bprop = zeros((insize2, outsize))
    jacobian_bprop_m_2_1(mod, sin1, sin2, sout, jac1_bprop, jac2_bprop)
    report_err(jac1_fprop, jac1_bprop, "jacobian input 1")
    report_err(jac2_fprop, jac2_bprop, "jacobian input 2")
    return (jac1_fprop, jac2_fprop, jac1_bprop, jac2_bprop)

def test_module_2_1_jac_param (mod, sin1, sin2, sout, minval=-2., maxval=2.):
    if not mod.has_params(): return
    sin1.x = sp.random.random(sin1.shape) * (maxval - minval) - minval
    sin2.x = sp.random.random(sin2.shape) * (maxval - minval) - minval
    for state in mod.parameter.states:
        state.x = sp.random.random(state.shape) * (maxval - minval) - minval
    mod.fprop(sin1, sin2, sout)
    insize  = mod.parameter.size()
    outsize = sout.size
    jac_fprop = zeros((insize, outsize))
    jacobian_fprop_m_2_1_param(mod, sin1, sin2, sout, jac_fprop)
    jac_bprop = zeros((insize, outsize))
    jacobian_bprop_m_2_1_param(mod, sin1, sin2, sout, jac_bprop)
    report_err(jac_fprop, jac_bprop, "jacobian param")
    return (jac_fprop, jac_bprop)

def make_test_m11_jac(ctor):
    def test_jac(size, *args, **kwargs):
        sin  = state(size)
        sout = state(())
        mod  = ctor(size, *args, **kwargs)
        mod.fprop(sin, sout) # resize sout
        test_module_1_1_jac(mod, sin, sout)
        test_module_1_1_jac_param(mod, sin, sout)
    return test_jac

def make_test_m21_jac(ctor):
    def test_jac(size1, size2=None, *args, **kwargs):
        if size2 is None: size2=size1
        sin1 = state(size1)
        sin2 = state(size2)
        sout = state(())
        mod  = ctor(size1, size2, *args, **kwargs)
        mod.fprop(sin1, sin2, sout) # resize out
        test_module_2_1_jac(mod, sin1, sin2, sout)
        test_module_2_1_jac_param(mod, sin1, sin2, sout)
    return test_jac



def test_layers_jac(*shapes):
    lins = [apply(linear, args) for args in zip(shapes, shapes[1:])]
    mod  = layers(*lins)
    sin  = state(shapes[0])
    sout = state(shapes[-1])
    test_module_1_1_jac(mod, sin, sout)
    test_module_1_1_jac_param(mod, sin, sout)

ctor_ns1 = lambda ctor: lambda s1,     *args, **kwargs: ctor(*args, **kwargs)
ctor_ns2 = lambda ctor: lambda s1, s2, *args, **kwargs: ctor(*args, **kwargs)

test_linear_jac        = make_test_m11_jac(linear)
test_bias_jac          = make_test_m11_jac(bias)
test_diag_jac          = make_test_m11_jac(diagonal)
test_convolution_jac   = make_test_m11_jac(ctor_ns1(convolution))
test_back_convolution_jac = make_test_m11_jac(ctor_ns1(back_convolution))
    
test_tanh_jac          = make_test_m11_jac(ctor_ns1(transfer_tanh))
test_abs_jac           = make_test_m11_jac(ctor_ns1(transfer_abs))
test_copy_flipsign_jac = make_test_m11_jac(ctor_ns1(transfer_copy_flipsign))
test_greater_jac       = make_test_m11_jac(ctor_ns1(transfer_greater))
test_double_abs_jac    = make_test_m11_jac(ctor_ns1(transfer_double_abs))


test_distance_l2_jac   = make_test_m21_jac(ctor_ns2(distance_l2))
test_crossent_jac      = make_test_m21_jac(ctor_ns2(cross_entropy))
test_penalty_l1_jac    = make_test_m11_jac(ctor_ns1(penalty_l1))

def test_jac():
    print '##########################################'
    print 'TEST LINEAR JACOBIAN'
    test_linear_jac( (2,5,5), (30,1,1) )
    print '##########################################'
    print 'TEST BIAS JACOBIAN'
    test_bias_jac( (2,5,5) )
    print '##########################################'
    print 'TEST BIAS JACOBIAN (per-feature)'
    test_bias_jac( (2,5,5), per_feature = True )
    print '##########################################'
    print 'TEST DIAGONAL JACOBIAN'
    test_diag_jac( (5,2,5) )
    print '##########################################'
    print 'TEST LAYERS JACOBIAN'
    test_layers_jac( (5,), (3,), (5,), (3,) )
    print '##########################################'
    print 'TEST TANH JACOBIAN'
    test_tanh_jac( (20,3,4) )
    print '##########################################'
    print 'TEST ABS JACOBIAN'
    test_abs_jac( (20,3,4) )
    print '##########################################'
    print 'TEST COPY-FLIPSIGN JACOBIAN'
    test_copy_flipsign_jac( (20,3,4) )
    print '##########################################'
    print 'TEST GREATER JACOBIAN'
    test_greater_jac( (20,3,4) )
    print '##########################################'
    print 'TEST DOUBLE-ABS JACOBIAN'
    test_double_abs_jac( (20,3,4) )
    print '##########################################'
    print 'TEST DISTANCE-L2 JACOBIAN'
    test_distance_l2_jac( (23,4,6) )
    print '##########################################'
    print 'TEST PENALTY-L1 JACOBIAN'
    test_penalty_l1_jac( (23,4,6) )
    print '##########################################'
    print 'TEST CROSS ENTROPY JACOBIAN'
    test_crossent_jac( (10,4,6) )
    print '##########################################'
    print 'TEST CONVOLUTION JACOBIAN'
    test_convolution_jac( (2,40,20), (5,7), convolution.full_table(2,1) )
    print '##########################################'
    print 'TEST BACK CONVOLUTION JACOBIAN'
    test_back_convolution_jac( (2,40,20), (5,7),
                               convolution.full_table(2,1) )

if __name__ == '__main__':
    test_jac()
