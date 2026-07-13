FC=f95
#
FFLAGS=-O  -fallow-argument-mismatch -m64
F90FLAGS=-O  -fallow-argument-mismatch -m64
#
.SUFFIXES:.f90 .f .o
.f.o:
	$(FC) $(FFLAGS) -c $<
.f90.o:
	$(FC) $(FFLAGS) -c $<
#
all: wimp_s
#
wimp_s: wimp_web.o graph.o
	$(FC) $(FFLAGS) -fno-backslash -o wimp wimp_web.o graph.o -L/usr/lib64 -L/usr/local/pgplot/pgplot_src -lpgplot -L/usr/lib64/ -lX11 -lpng
	chmod a+rx wimp
#
clean: 
	rm *.o
