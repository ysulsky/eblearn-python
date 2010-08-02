#!/usr/bin/env python

from eblearn import *

# test (g(f))'' instead of f'' to ensure a non-zero second derivative
test_xfer = transfer_square # |None | transfer_square | transfer_cube

# Note: since bbprop assumes that the off-diagonal elements of the
# Hessian matrix are zero, any bbprop test for a machine with
# nonzero mixed second partial derivatives might fail

dx, ddx      = 1e-6, 1e-4
jacobian_tol = 0.01, 1e-6 # relative, actual
hessian_tol  = 0.05, 1e-5 

def print_testing(name):
    print '#' * 50
    print '# TESTING ' + name

def report_err (a, b, name, tol):
    def report_str_err(err1, err2, s):
        passed = err1 < tol[0] or err2 < tol[1]
        err1 = '%.4g' % err1
        err2 = '(%.4g)' % err2
        e = ' = %-12s %-12s %s' % (err1, err2, "pass" if passed else "FAIL")
        if not passed: print '*' * 80
        print '%-45s%-35s' % (s, e)
        if not passed: print '*' * 80
    d = a-b
    rel_err = 2 * (thresh_less(d, abs(d), 1e-6))/((a+b) + 1e-6)
    max_rel_err = abs(rel_err).max()
    max_act_err = abs(d).max()
    tot_rel_err = sqrt(sumsq(rel_err))
    tot_act_err = sqrt(sumsq(d))
    report_str_err(max_rel_err, max_act_err,
                   'Max %s rel. (abs.) error' % (name,))
    #report_str_err(max_err, 'Total %s relative difference' % (name,))

def clear_state(s):
    s.clear_dx()
    s.clear_ddx()

def clear_param(p):
    p.clear_dx()
    p.clear_ddx()

def rand_param(p, rmin, rmax):
    for state in p.states:
        state.x[:]=random(state.shape) * (rmax - rmin) + rmin

def mod_bprop(snd, mod, *args):
    mod.bprop(*args)
    if snd: mod.bbprop(*args)

mod_bprop11 = mod_bprop21 = mod_bprop

def jacobian_bwd_m_1_1 (mod, sin, sout, snd=False):
    jac = zeros((sin.size, sout.size))
    for i in xrange(sout.size):
        clear_state(sin)
        clear_state(sout)
        sout.dx.flat[i] = 1.
        mod_bprop11(snd, mod, sin, sout)
        jac[:,i] = (sin.ddx if snd else sin.dx).ravel()
    return jac

def jacobian_bwd_m_1_1_param (mod, sin, sout, snd=False):
    jac = zeros((mod.parameter.size(), sout.size))
    param    = mod.parameter
    param_dx = lambda: np.fromiter((param.ddx if snd else param.dx), rtype)
    for i in xrange(sout.size):
        clear_state(sout)
        sout.dx.flat[i] = 1.
        clear_param(param)
        mod_bprop11(snd, mod, sin, sout)
        jac[:,i] = param_dx()
    return jac

def jacobian_fwd_m_1_1 (mod, sin, sout, snd=False):
    jac = zeros((sin.size, sout.size))
    small = ddx if snd else dx
    sina  = state(sin.shape);     souta = state(sout.shape)
    sinb  = state(sin.shape);     soutb = state(sout.shape)
    sin_x = sin.x.ravel();
    sina_x = sina.x.ravel(); sinb_x = sinb.x.ravel();
    sina_x[:] = sin_x;       sinb_x[:] = sin_x;
    
    if snd:
        mod.fprop(sin, sout)
    
    for i in xrange(sin.size):
        sina_x[i] -= small
        mod.fprop(sina, souta)
        sina_x[i]  = sin_x[i]
        
        sinb_x[i] += small
        mod.fprop(sinb, soutb)
        sinb_x[i]  = sin_x[i]

        if snd:
            jac[i,:] = ((soutb.x - 2*sout.x + souta.x) / (small**2)).ravel()
        else:
            jac[i,:] = ((soutb.x - souta.x) / (2 * small)).ravel()
    return jac

def jacobian_fwd_m_1_1_param (mod, sin, sout, snd=False):
    jac = zeros((mod.parameter.size(), sout.size))
    small = ddx if snd else dx
    souta = state(sout.shape)
    soutb = state(sout.shape)
    i = -1
    if snd:
        mod.fprop(sin, sout)
    for pstate in mod.parameter.states:
        for j in xrange(pstate.size):
            i += 1
            pstate_x = pstate.x.ravel()

            pstate_x[j] -= small
            mod.fprop(sin, souta)

            pstate_x[j] += 2 * small
            mod.fprop(sin, soutb)

            pstate_x[j] -= small
            if snd:
                jac[i,:] = ((soutb.x - 2*sout.x + souta.x) / (small**2)).ravel()
            else:
                jac[i,:] = ((soutb.x - souta.x) / (2 * small)).ravel()
    return jac


def test_module_1_1_jac (mod, sin, sout, skip_bbprop=False):
    mod.fprop(sin, sout)
    jac_fprop = jacobian_fwd_m_1_1(mod, sin, sout)
    jac_bprop = jacobian_bwd_m_1_1(mod, sin, sout)
    report_err(jac_fprop, jac_bprop, "jacobian     input", jacobian_tol)
    if skip_bbprop:
        print '   *** skipping bbprop test ***   '
        return (jac_fprop, jac_bprop)
    if test_xfer: mod = layers(mod, test_xfer())
    hes_fprop = jacobian_fwd_m_1_1(mod, sin, sout, True)
    hes_bprop = jacobian_bwd_m_1_1(mod, sin, sout, True)
    report_err(hes_fprop, hes_bprop, "diag hessian input", hessian_tol)
    return (jac_fprop, jac_bprop, hes_fprop, hes_bprop)

def test_module_1_1_jac_param (mod, sin, sout, skip_bbprop=False):
    if not mod.has_params(): return
    mod.fprop(sin, sout)
    jac_fprop = jacobian_fwd_m_1_1_param(mod, sin, sout)
    jac_bprop = jacobian_bwd_m_1_1_param(mod, sin, sout)
    report_err(jac_fprop, jac_bprop, "jacobian     param", jacobian_tol)
    if skip_bbprop:
        print '   *** skipping bbprop test ***   '
        return (jac_fprop, jac_bprop)
    if test_xfer   : mod = layers(mod, test_xfer())
    hes_fprop = jacobian_fwd_m_1_1_param(mod, sin, sout, True)
    hes_bprop = jacobian_bwd_m_1_1_param(mod, sin, sout, True)
    report_err(hes_fprop, hes_bprop, "diag hessian param", hessian_tol)
    return (jac_fprop, jac_bprop, hes_fprop, hes_bprop)


def jacobian_bwd_m_2_1 (mod, sin1, sin2, sout, snd=False):
    jac1 = zeros((sin1.size, sout.size))
    jac2 = zeros((sin2.size, sout.size))
    for i in xrange(sout.size):
        clear_state(sin1)
        clear_state(sin2)
        clear_state(sout)
        sout.dx.flat[i] = 1.
        mod_bprop21(snd, mod, sin1, sin2, sout)
        jac1[:,i] = (sin1.ddx if snd else sin1.dx).ravel()
        jac2[:,i] = (sin2.ddx if snd else sin2.dx).ravel()        
    return jac1, jac2

def jacobian_bwd_m_2_1_param (mod, sin1, sin2, sout, snd=False):
    jac = zeros((mod.parameter.size(), sout.size))
    param    = mod.parameter
    param_dx = lambda: np.fromiter((param.ddx if snd else param.dx), rtype)
    for i in xrange(sout.size):
        clear_state(sout)
        sout.dx.flat[i] = 1.
        clear_param(param)
        mod_bprop21(snd, mod, sin1, sin2, sout)
        jac[:,i] = param_dx()
    return jac

def jacobian_fwd_m_2_1 (mod, sin1, sin2, sout, snd=False):
    jac1 = zeros((sin1.size, sout.size))
    jac2 = zeros((sin2.size, sout.size))
    small = ddx if snd else dx
    sins = [sin1, sin2]; jacs = [jac1, jac2]
    if snd: mod.fprop(sin1, sin2, sout)
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

            if snd:
                jac[i,:] = ((soutb.x - 2*sout.x + souta.x) / (small**2)).ravel()
            else:
                jac[i,:] = ((soutb.x - souta.x) / (2 * small)).ravel()
    return jac1, jac2


def jacobian_fwd_m_2_1_param (mod, sin1, sin2, sout, snd=False):
    jac = zeros((mod.parameter.size(), sout.size))
    small = ddx if snd else dx
    souta = state(sout.shape)
    soutb = state(sout.shape)
    i = -1
    if snd:
        mod.fprop(sin1, sin2, sout)
    for pstate in mod.parameter.states:
        for j in xrange(pstate.size):
            i += 1
            pstate_x = pstate.x.ravel()

            pstate_x[j] -= small
            mod.fprop(sin1, sin2, souta)

            pstate_x[j] += 2 * small
            mod.fprop(sin1, sin2, soutb)

            pstate_x[j] -= small
            if snd:
                jac[i,:] = ((soutb.x - 2*sout.x + souta.x) / (small**2)).ravel()
            else:
                jac[i,:] = ((soutb.x - souta.x) / (2 * small)).ravel()
    return jac

def test_module_2_1_jac (mod, sin1, sin2, sout, skip_bbprop=False):
    mod.fprop(sin1, sin2, sout)
    jac1_fprop, jac2_fprop = jacobian_fwd_m_2_1(mod, sin1, sin2, sout)
    jac1_bprop, jac2_bprop = jacobian_bwd_m_2_1(mod, sin1, sin2, sout)
    report_err(jac1_fprop, jac1_bprop, "jacobian     input 1", jacobian_tol)
    report_err(jac2_fprop, jac2_bprop, "jacobian     input 2", jacobian_tol)
    if skip_bbprop:
        print '   *** skipping bbprop test ***   '
        return (jac1_fprop, jac2_fprop, jac1_bprop, jac2_bprop,)
    if test_xfer: mod = filter_output_2_1(mod, test_xfer())
    hes1_fprop, hes2_fprop = jacobian_fwd_m_2_1(mod, sin1, sin2, sout, True)
    hes1_bprop, hes2_bprop = jacobian_bwd_m_2_1(mod, sin1, sin2, sout, True)
    report_err(hes1_fprop, hes1_bprop, "diag hessian input 1", hessian_tol)
    report_err(hes2_fprop, hes2_bprop, "diag hessian input 2", hessian_tol)
    return (jac1_fprop, jac2_fprop, jac1_bprop, jac2_bprop,
            hes1_fprop, hes2_fprop, hes1_bprop, hes2_bprop)

def test_module_2_1_jac_param (mod, sin1, sin2, sout, skip_bbprop=False):
    if not mod.has_params(): return
    mod.fprop(sin1, sin2, sout)
    jac_fprop = jacobian_fwd_m_2_1_param(mod, sin1, sin2, sout)
    jac_bprop = jacobian_bwd_m_2_1_param(mod, sin1, sin2, sout)
    report_err(jac_fprop, jac_bprop,   "jacobian     param", jacobian_tol)
    if skip_bbprop:
        print '   *** skipping bbprop test ***   '
        return (jac_fprop, jac_bprop)
    if test_xfer: mod = filter_output_2_1(mod, test_xfer())
    hes_fprop = jacobian_fwd_m_2_1_param(mod, sin1, sin2, sout, True)
    hes_bprop = jacobian_bwd_m_2_1_param(mod, sin1, sin2, sout, True)
    report_err(hes_fprop, hes_bprop,   "diag hessian param", hessian_tol)
    return (jac_fprop, jac_bprop, hes_fprop, hes_bprop)

def make_test_m11_jac(ctor):
    def test_jac(size, *args, **kwargs):
        name = kwargs.pop('name', None)
        sin  = kwargs.pop('use_input', None)
        mod  = kwargs.pop('use_mod', None)
        rmin, rmax  = kwargs.pop('rrange', (-2, 2))
        skip_bbprop = kwargs.pop('skip_bbprop', False)
        sout = state(())
        if sin is None:
            sin = state(size)
            sin.x[:] = random(sin.shape) * (rmax - rmin) + rmin
        else:
            size = sin.shape
        if mod is None:
            mod  = ctor(size, *args, **kwargs)
            rand_param(mod.parameter, rmin, rmax)
        name = name or mod.__class__.__name__
        print_testing(name)
        mod.fprop(sin, sout) # resize sout
        r1 = test_module_1_1_jac(mod, sin, sout, skip_bbprop)
        r2 = test_module_1_1_jac_param(mod, sin, sout, skip_bbprop)
        return (mod, sin, r1, r2)
    return test_jac

def make_test_m21_jac(ctor):
    def test_jac(size1, size2=None, *args, **kwargs):
        if size2 is None: size2=size1
        name = kwargs.pop('name', None)
        mod  = kwargs.pop('use_mod', None)
        sin1 = kwargs.pop('use_input1', None)
        sin2 = kwargs.pop('use_input2', None)
        rmin, rmax  = kwargs.pop('rrange', (-2, 2))
        skip_bbprop = kwargs.pop('skip_bbprop', False)
        sout = state(())
        if sin1 is None:
            sin1 = state(size1)
            sin1.x[:] = random(sin1.shape) * (rmax - rmin) + rmin
        else:
            size1 = sin.shape
        if sin2 is None:
            sin2 = state(size2)
            sin2.x[:] = random(sin2.shape) * (rmax - rmin) + rmin
        else:
            size2 = sin.shape
        if mod is None:
            mod  = ctor(size1, size2, *args, **kwargs)
            rand_param(mod.parameter, rmin, rmax)
        name = name or mod.__class__.__name__
        print_testing(name)
        mod.fprop(sin1, sin2, sout) # resize out
        r1 = test_module_2_1_jac(mod, sin1, sin2, sout, skip_bbprop)
        r2 = test_module_2_1_jac_param(mod, sin1, sin2, sout, skip_bbprop)
        return (mod, sin1, sin2, r1, r2)
    return test_jac

def test_layers_1_jac(nlayers, insize, outsize, **kwargs):
    ftest = make_test_m11_jac(None)
    mod  = kwargs.get('use_mod')
    if mod is None:
        layer1 = linear(insize, outsize)
        rest = [transfer_identity() for i in range(nlayers-1)]
        mod  = layers_1(layer1, *rest)
        kwargs['use_mod'] = mod
    return ftest(insize, **kwargs)


ctor_ns1 = lambda ctor: lambda s1,     *args, **kwargs: ctor(*args, **kwargs)
ctor_ns2 = lambda ctor: lambda s1, s2, *args, **kwargs: ctor(*args, **kwargs)

test_linear_jac        = make_test_m11_jac(linear)
test_bias_jac          = make_test_m11_jac(bias_module)
test_diag_jac          = make_test_m11_jac(diagonal)
test_mult_jac          = make_test_m21_jac(ctor_ns2(multiplication))
test_convolution_jac   = make_test_m11_jac(ctor_ns1(convolution))
test_back_convolution_jac = make_test_m11_jac(ctor_ns1(back_convolution))

test_identity_jac      = make_test_m11_jac(ctor_ns1(transfer_identity))
test_square_jac        = make_test_m11_jac(ctor_ns1(transfer_square))
test_cube_jac          = make_test_m11_jac(ctor_ns1(transfer_cube))
test_exp_jac           = make_test_m11_jac(ctor_ns1(transfer_exp))
test_tanh_jac          = make_test_m11_jac(ctor_ns1(transfer_tanh))
test_abs_jac           = make_test_m11_jac(ctor_ns1(transfer_abs))
test_copy_flipsign_jac = make_test_m11_jac(ctor_ns1(transfer_copy_flipsign))
test_greater_jac       = make_test_m11_jac(ctor_ns1(transfer_greater))
test_double_abs_jac    = make_test_m11_jac(ctor_ns1(transfer_double_abs))


test_distance_l2_jac   = make_test_m21_jac(ctor_ns2(distance_l2))
test_bconv_rec_cost_jac= make_test_m21_jac(ctor_ns2(bconv_rec_cost))
test_crossent_jac      = make_test_m21_jac(ctor_ns2(cross_entropy))
test_penalty_l1_jac    = make_test_m11_jac(ctor_ns1(penalty_l1))
test_penalty_l2_jac    = make_test_m11_jac(ctor_ns1(penalty_l2))

def _test_jac():
    test_linear_jac( (2,2,2), (3,1,1) )
    test_bias_jac( (2,5,5) )
    test_bias_jac( (2,5,5), per_feature = True, name='bias (per feature)' )
    test_diag_jac( (5,2,5) )
    test_mult_jac( (5,2,5) )
    test_tanh_jac( (2,3,4), skip_bbprop=True )
    test_identity_jac( (2,3,4) )
    test_square_jac( (2,3,4) )
    test_cube_jac( (2,3,4) )
    test_exp_jac( (2,3,4) )
    test_abs_jac( (2,3,4) )
    test_copy_flipsign_jac( (2,3,4) )
    test_greater_jac( (2,3,4) )
    test_double_abs_jac( (2,3,4) )
    test_distance_l2_jac( (2,1,1) )
    test_distance_l2_jac( (2,1,1), average=False, name='distance_l2 (no avg)' )
    test_bconv_rec_cost_jac( (4,2,2),
                             coeff=bconv_rec_cost.coeff_from_conv((4,2,2),
                                                                  (3,1,2) ))
    test_penalty_l1_jac( (2,1,1) )
    test_penalty_l1_jac( (2,1,1), average=False, name='penalty_l1 (no avg)')
    test_penalty_l2_jac( (2,1,1) )
    test_penalty_l2_jac( (2,1,1), average=False, name='penalty_l2 (no avg)')
    test_crossent_jac( (10,1,1) )
    test_convolution_jac( (2,3,4), (2,3), convolution.full_table(2,1) )
    test_back_convolution_jac( (2,3,4), (2,3),
                               convolution.full_table(2,1) )
    test_layers_1_jac(1, (2,1,1), (1,2,2))

def test_jac():
    # disable IPP
    correlate.eblearn_disable_ipp()
    try:
        _test_jac()
    finally:
        correlate.eblearn_enable_ipp()

if __name__ == '__main__':
    test_jac()

