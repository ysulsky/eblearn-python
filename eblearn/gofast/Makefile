CC      = gcc
CFLAGS  = -g -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing
LDFLAGS = -fPIC -shared
CYTHON  = cython

INCLUDES = $(shell python-config --includes) \
           $(shell ./numpy_includes)

# optional IPP support -- enter a non-empty IPP_DIR to enable
IPP_DIR = #/opt/intel/ipp/5.3.1.062/em64t
IPP_ARCH    = em64t
IPP_CFLAGS  = -I$(IPP_DIR)/include
IPP_LDFLAGS = -L$(IPP_DIR)/lib
IPPI_LDFLAGS = -lippiemerged$(IPP_ARCH) -lippimerged$(IPP_ARCH)
IPPS_LDFLAGS = -lippsemerged$(IPP_ARCH) -lippsmerged$(IPP_ARCH)
IPPCORE_LDFLAGS = -lippcore$(IPP_ARCH)

PYX_EXTS = util vecmath idx correlate

ifneq (,$(IPP_DIR))
PYX_EXTS += ipp
CFLAGS_ipp  = $(IPP_CFLAGS)
LDFLAGS_ipp = $(IPP_LDFLAGS) $(IPPI_LDFLAGS) $(IPPS_LDFLAGS) $(IPPCORE_LDFLAGS)
endif

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
	$(CC) -c $(CFLAGS) $(INCLUDES) $(CFLAGS_$*) $*.c

$(addsuffix .so, $(PYX_EXTS)) : %.so : %.o
	$(CC) $*.o $(LDFLAGS) $(LDFLAGS_$*) -o $@


