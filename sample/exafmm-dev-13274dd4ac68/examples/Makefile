# Usage:
# $ env CC=icc CXX=icpc make VAR=VAL tapas
#
# NOTE: "make CC=icc CXX=icpc ..." does not work.

include ../Makefile.include

#LFLAGS	+= -DMASS # Use all positive sources
#LFLAGS	+= -DIneJ # Different target and source points
LFLAGS	+= -DDIRECT # Compare with direct summation
LFLAGS	+= -DWRITE_TIME # Write timings to file

MODE ?= debug
ifeq ($(MODE),debug)
	OPT = -O0
	DEBUG = -DTAPAS_DEBUG=1 -g -gdwarf-2
	LOG = -DTAPAS_LOG_LEVEL=1
else
	OPT = -O3 # -O3 is also OK, but when gprof is used, gprof command misunderstand names of functions called as 'frame_dummy'
	DEBUG = -DTAPAS_DEBUG=0
	LOG = -DTAPAS_LOG_LEVEL=0
endif

# Test for tree construction
tree: tree.o $(OBJECTS)
	$(CXX) $? $(LFLAGS)
	./a.out

# Test for serial FMM
serial: serial.o $(OBJECTS)
	$(CXX) $? $(LFLAGS) -o $@

# Test for parallel FMM
parallel: parallel.cxx $(OBJECTS)
	$(MPICXX) $? $(LFLAGS) -o $@
#	mpirun -np 9 ./a.out --ncrit 256 --distribution plummer

tapas: serial_tapas parallel_tapas

serial_tapas: tapas_exafmm.cxx $(OBJECTS)
	$(CXX) $(DEBUG) $(LOG) $^ $(LFLAGS) -o $@

parallel_tapas: tapas_exafmm.cxx $(OBJECTS)
	$(MPICXX) $(DEBUG) $(LOG) -DUSE_MPI $^ $(LFLAGS) -o $@ 

# Checking O(N) complexity
complexity: serial.o $(OBJECTS)
	$(CXX) $? $(LFLAGS)
	for N in 10000 20000 50000 100000 200000 500000 1000000; do \
        ./a.out --numBodies $$N; echo; \
	done

# Test for kernels only
kernel: kernel.o $(OBJECTS)
	$(CXX) $? $(LFLAGS)
	./a.out

# Checking O(theta^p) convergence (Turn on VTK to see a graph at the end)
convergence: kernel.cxx $(SOURCES)
	rm -f kernel.dat
	for P in 3 4 5 6 7 8 9 10 11 12 13 14 15; do echo P = $$P compiling && \
	$(CXX) $? $(LFLAGS) -DEXPANSION=$$P && \
	echo P = $$P executing && ./a.out; done

# Ewald vs. periodic FMM
ewald: ewald.o $(OBJECTS)
	$(CXX) $? $(LFLAGS)
	mpirun -np 2 ./a.out --numBodies 1000 --ncrit 32 --images 3

