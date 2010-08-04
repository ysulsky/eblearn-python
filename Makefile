all:
	cd eblearn; $(MAKE) $(MFLAGS) all
	cd demos;   $(MAKE) $(MFLAGS) all
clean:
	cd eblearn; $(MAKE) $(MFLAGS) clean
	cd demos;   $(MAKE) $(MFLAGS) clean

