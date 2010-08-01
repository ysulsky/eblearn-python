CC = gcc
CFLAGS = -g -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing
LDFLAGS= -fPIC -shared
CYTHON = cython

INCLUDES = $(shell python-config --includes) \
           $(shell ./numpy_includes)

PYX_EXTS = _util _vecmath _idx _correlate
PYX_DEPS = $(addsuffix .d, $(PYX_EXTS))
ALLDEPS  = $(PYX_DEPS)

all: depend $(addsuffix .so, $(PYX_EXTS))
depend: $(ALLDEPS)
clean:
	rm -f *.pyc
	rm -f $(addsuffix .o,  $(PYX_EXTS))
	rm -f $(addsuffix .so, $(PYX_EXTS))
	rm -f $(addsuffix .c,  $(PYX_EXTS))
	rm -f $(ALLDEPS)

-include $(ALLDEPS)

$(PYX_DEPS) : %.d : %.pyx
	@echo $*.o $*.d : $(shell ./pyx_deps $*) > $@

$(addsuffix .o, $(PYX_EXTS)) : %.o : %.pyx
	$(CYTHON) $*.pyx
	$(CC) -c $(CFLAGS) $(INCLUDES) $*.c

$(addsuffix .so, $(PYX_EXTS)) : %.so : %.o
	$(CC) $(LDFLAGS) $*.o -o $@

