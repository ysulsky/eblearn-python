all:
	cd eblearn; $(MAKE) $(MFLAGS) all
clean:
	cd eblearn; $(MAKE) $(MFLAGS) clean
	rm -rf demo/outputs; rm -rf demo/exper*

