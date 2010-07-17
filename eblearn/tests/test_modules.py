from eblearn import *

def report_err (a, b, name, tol = 1e-5):
    def report_str_err(err, s):
        print '%-40s = %-15g %15s' % (s, err, "pass" if err < tol else "FAIL")
    max_err = abs(a - b).max()
    tot_err = sqrt(sqdist(a, b))
    report_str_err(max_err, 'Max   %s distance' % name)
    report_str_err(tot_err, 'Total %s distance' % name)

def jacobian_bprop_m_1_1 (mod, sin, sout, jac):
    for i in xrange(sout.size):
        sout.dx.fill(0.)
        sin.dx.fill(0.)
        sout.dx.flat[i] = 1.
        mod.bprop_input(sin, sout)
        jac[:,i] = sin.dx.flat

def jacobian_bprop_m_1_1_param (mod, sin, sout, jac):
    for i in xrange(sout.size):
        sout.dx.fill(0.)
        mod.parameter.clear_dx()
        sout.dx.flat[i] = 1.
        mod.bprop_param(sin, sout)
        jac[:,i] = sp.fromiter(mod.parameter.dx, rtype)

def jacobian_fprop_m_1_1 (mod, sin, sout, jac):
    small = 1e-6
    sina  = state(sin.shape)
    sinb  = state(sin.shape)
    souta = state(sout.shape)
    soutb = state(sout.shape)
    for i in xrange(sin.size):
        sina.x[:] = sin.x
        sina.x.flat[i] -= small
        mod.fprop(sina, souta)
        
        sinb.x[:] = sin.x
        sinb.x.flat[i] += small
        mod.fprop(sinb, soutb)
        
        jac[i,:] = ((soutb.x - souta.x) / (2 * small)).flat

def jacobian_fprop_m_1_1_param (mod, sin, sout, jac):
    small = 1e-6
    souta = state(sout.shape)
    soutb = state(sout.shape)
    i = -1
    for pstate in mod.parameter.states:
        for j in xrange(pstate.size):
            i += 1

            pstate.x.flat[j] -= small
            mod.fprop(sin, souta)

            pstate.x.flat[j] += 2 * small
            mod.fprop(sin, soutb)

            pstate.x.flat[j] -= small
            jac[i,:] = ((soutb.x - souta.x) / (2 * small)).flat

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
    (jac_fprop, jac_bprop)

def test_module_1_1_jac_param (mod, sin, sout, minval=-2., maxval=2.):
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

def test_linear_jac(fro, to):
    mod  = linear(fro, to)
    sin  = state(fro)
    sout = state(to)
    test_module_1_1_jac(mod, sin, sout)
    test_module_1_1_jac_param(mod, sin, sout)

def test_bias_jac(size):
    mod  = bias(size)
    sin  = state(size)
    sout = state(size)
    test_module_1_1_jac(mod, sin, sout)
    test_module_1_1_jac_param(mod, sin, sout)
    

def test_jac():
    print '##########################################'
    print 'TEST LINEAR JACOBIAN'
    test_linear_jac( (2,5,5), (30,1,1) )
    print '##########################################'
    print 'TEST BIAS JACOBIAN'
    test_bias_jac( (2,5,5) )

if __name__ == '__main__':
    test_jac()
