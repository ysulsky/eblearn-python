import numpy as np
from ._vecmath import *

__all__ = ['clear',
           'sqmag', 'sqdist', 'dtanh', 'ddtanh',
           'ldot', 'm1ldot', 'm2ldot', 'm3ldot', 'mdotc',
           'm2kdotmk', 'm2dotm1', 'm4dotm2', 'm6dotm3',
           'mkextmk', 'm1extm1', 'm2extm2', 'm3extm3',
           'm2dotrows', 'normrows']

def copy_normrows(m): # for testing
    x=m.copy()
    normrows(x)
    return x

