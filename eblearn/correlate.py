from eblearn.util import replace_global, rtype
from numpy        import float32

# The public interface is via the [config_][back_]<correlate|convolve>[_table]
# functions
__all__      = [meta + prefix + fn + suffix
                for meta   in ('config_', '')
                for prefix in ('back_',   '')
                for fn     in ('correlate', 'convolve')
                for suffix in ('_table', '')]

# Implementations may override individual pieces such as "m2_correlate_table"
__all_impl__ = [dim + prefix + fn + suffix
                for dim    in (['gen_']+['m%d_' % d for d in range(1,4)])
                for prefix in ('back_', '')
                for fn     in ('correlate', 'convolve')
                for suffix in ('_table', '')]


import eblearn.goslow.correlate as slow_ver

fast_ver_enabled = True
try:
    import eblearn.gofast.correlate as fast_ver
except ImportError:
    fast_ver = None
    fast_ver_enabled = False

ipp_ver_enabled = True
try:
    import eblearn.gofast.ipp as ipp_ver
except ImportError:
    ipp_ver = None
    ipp_ver_enabled = False

theano_ver_enabled = True
try:
    import eblearn.gofast.theano as theano_ver
except ImportError:
    theano_ver = None
    theano_ver_enabled = False
                
all_vers = dict(slow_ver   = slow_ver,
                fast_ver   = fast_ver,
                ipp_ver    = ipp_ver,
                theano_ver = theano_ver)

def reset_implementations(packages = ('eblearn',)):
    mod_globals = globals()
    def use_ver(ver):
        if ver is None: return
        assert (ver in all_vers.values()), "forgot to update all_vers"
        updates = [(k, getattr(ver, k)) for k in __all_impl__ + __all__ 
                   if  hasattr(ver, k)]
        mod_globals.update(updates)
    
    use_ver(slow_ver)
    if fast_ver_enabled: 
        use_ver(fast_ver)
    
    # theano only does has GPU implementations for rtype = float32
    # so prefer IPP otherwise
    if rtype == float32:
        if ipp_ver_enabled:    use_ver(ipp_ver)
        if theano_ver_enabled: use_ver(theano_ver)
    else:
        if theano_ver_enabled: use_ver(theano_ver)
        if ipp_ver_enabled:    use_ver(ipp_ver)
    
    # update the vtables
    correlate_vtable = dict([(k, mod_globals[k]) 
                             for k in __all_impl__ + __all__])
    for ver in all_vers.values():
        if ver is None: continue
        if hasattr(ver, 'set_correlate_module_vtable'):
            ver.set_correlate_module_vtable(correlate_vtable)
    
    # update modules that have done "from correlate import ..."
    if packages:
        for k in __all__:
            new_fn  = mod_globals[k]
            old_fns = set(getattr(ver, k, new_fn) 
                          for ver in all_vers.values() if ver is not None)
            old_fns.remove(new_fn)
            for old_fn in old_fns:
                replace_global(old_fn, new_fn, packages)


reset_implementations(packages=None)

