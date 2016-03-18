
MODE ?= release

ifeq ($(MODE),debug)
	OPT = -O0
	DEBUG = -DTAPAS_DEBUG=1 -g -gdwarf-2
	LOG = -DTAPAS_LOG_LEVEL=1
else
	OPT = -O3 # -O3 is also OK, but when gprof is used, gprof command misunderstand names of functions called as 'frame_dummy'
	DEBUG = -DTAPAS_DEBUG=0
	LOG = -DTAPAS_LOG_LEVEL=0
endif

CC ?= clang
CXX ?= clang++
MPICXX ?= mpicxx
CXXFLAGS := -std=c++11 -Wall -Wextra -fno-elide-constructors

#PROF := -pg

ifneq (,$(findstring clang++,$(CXX)))
	CXXFLAGS += -funroll-loops -fcolor-diagnostics -Wno-sign-compare -Wno-unused-function
else ifneq (, $(findstring g++, $(CXX)))
	CXXFLAGS += -funroll-loops -fdiagnostics-color=auto \
							-Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-function
else ifneq (, $(findstring icpc, $(CXX)))
endif

CXXFLAGS += -DS=$(S)

CXXFLAGS += -DTAPAS_BH # temporary cxxflag for Tapas Barnes-Hut implementation.

ifeq ($(LET),hardcoded)
	CXXFLAGS += -DOLD_LET_TRAVERSE
endif

ifeq ($(LET),manual)
	CXXFLAGS += -DMANUAL_LET
endif

ifeq ($(LET),auto-slow)
	CXXFLAGS += -DAUTO_LET_SLOW
endif

TAPAS_DIR = ../../cpp

all: bh bh_mpi

bh.o: bh.cc
	$(CXX) $(CXXFLAGS) $(DEBUG) $(LOG) $(OPT) -c $^ -I$(TAPAS_DIR)/include

bh: bh.o
	$(CXX) $^ -o $@

bh_mpi.o: bh.cc
	$(MPICXX) -DUSE_MPI -o $@ $(CXXFLAGS)  -I$(TAPAS_DIR)/include $(OPT) $(DEBUG) $(LOG) -c $^ $(PROF)

bh_mpi: bh_mpi.o
	$(MPICXX) -DUSE_MPI $^ $(PROF) -o $@ $(PROF)

clean:
	$(RM) -f *.o bh bh_mpi
