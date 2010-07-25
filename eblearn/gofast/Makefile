CC=gcc
CFLAGS=-g -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing
INCLUDES=$(shell python-config --includes) $(shell python -c "from numpy.distutils.misc_util import *; print '-I'+' -I'.join(get_numpy_include_dirs())")

all: _util.so _idx.so _vecmath.so _correlate.so
clean:
	rm -f *.c *.so *.pyc

_util.so: _util.c
	$(CC) $(CFLAGS) $(INCLUDES) -o _util.so _util.c
_util.c: _util.pyx _util.pxd
	cython _util.pyx

_vecmath.so: _vecmath.c
	$(CC) $(CFLAGS) $(INCLUDES) -o _vecmath.so _vecmath.c
_vecmath.c: _vecmath.pyx _util.pxd
	cython _vecmath.pyx

_idx.so: _idx.c
	$(CC) $(CFLAGS) $(INCLUDES) -o _idx.so _idx.c
_idx.c: _idx.pyx _util.pxd
	cython _idx.pyx

_correlate.so: _correlate.c
	$(CC) $(CFLAGS) $(INCLUDES) -o _correlate.so _correlate.c
_correlate.c: _correlate.pyx _util.pxd
	cython _correlate.pyx



