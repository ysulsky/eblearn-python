from eblearn import *

def test_all():
    from . import test_modules
    test_modules.test()
    from . import test_vecmath
    test_vecmath.test()
    from . import test_correlate
    test_correlate.test()

