# Build using g++
# CXX = g++
# MPICXX = mpicxx -cxx=g++ -DUSE_MPI
# CXXFLAGS := -std=c++11 -Wall -Wextra -O0
# CXXFLAGS += -g -g3 -ggdb -Wall -fno-elide-constructors

MODE ?= debug
#MODE ?= release

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
MPICXX ?= mpicxx -cxx=$(CXX) -DUSE_MPI
CXXFLAGS := -std=c++11 -Wall -Wextra -fno-elide-constructors

#PROF := -pg

ifneq (,$(findstring clang++,$(CXX)))
	CXXFLAGS += -funroll-loops -fcolor-diagnostics -Wno-sign-compare
else ifneq (, $(findstring g++, $(CXX)))
	CXXFLAGS += -funroll-loops -Wno-sign-compare -fdiagnostics-color=auto -Wno-unknown-pragmas
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
	$(MPICXX) -o $@ $(CXXFLAGS)  -I$(TAPAS_DIR)/include $(OPT) $(DEBUG) $(LOG) -c $^ $(PROF)

bh_mpi: bh_mpi.o
	$(MPICXX) $^ $(PROF) -o $@ $(PROF)

clean:
	$(RM) -f *.o bh bh_mpi
