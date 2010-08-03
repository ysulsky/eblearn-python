
__all__ = ['clear',
           'sumabs', 'sumsq', 'sqdist', 'dtanh', 'ddtanh',
           'ldot', 'm1ldot', 'm2ldot', 'm3ldot', 'mdotc',
           'm2kdotmk', 'm2dotm1', 'm4dotm2', 'm6dotm3',
           'mkextmk', 'm1extm1', 'm2extm2', 'm3extm3',
           'm2dotrows', 'normrows', 'copy_normrows',
##### no gofast versions for these yet --
           'thresh_less']

import eblearn.goslow.vecmath as slow_ver
from   eblearn.goslow.vecmath import *

try:
    import eblearn.gofast.vecmath as fast_ver
    from   eblearn.gofast.vecmath import *
except ImportError: 
    fast_ver = None

