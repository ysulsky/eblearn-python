CC=gcc
CFLAGS=-shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing
INCLUDES=$(shell python-config --includes) $(shell python -c "from numpy.distutils.misc_util import *; print '-I'+' -I'.join(get_numpy_include_dirs())")

all: convolve.so
clean:
	rm -f *.c *.so

convolve.so: convolve.c
	$(CC) $(CFLAGS) $(INCLUDES) -o convolve.so convolve.c

convolve.c: convolve.pyx
	cython convolve.pyx

