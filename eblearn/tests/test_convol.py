from ..gofast.convolve import correlate_2d_valid, correlate_2d_valid_simple
from ..util import *

x = sp.lena().astype(rtype)
k = sp.random.random((10,10))

y1=correlate_2d_valid(x, k)
y2=correlate_2d_valid_simple(x, k)

assert (y1 == y2).all()

xt = x.T
y1t = correlate_2d_valid(x, k)
y2t = correlate_2d_valid_simple(x, k)

assert (y1t == y2t).all()


