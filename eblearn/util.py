import scipy as sp
from math import sqrt, log
from _util import rtype

zeros = lambda shape: sp.zeros(shape, rtype, 'C')
ones  = lambda shape: sp.ones(shape, rtype, 'C')
product = sp.prod

def sqdist(a, b):
    m = a - b
    return (m * m).sum()
