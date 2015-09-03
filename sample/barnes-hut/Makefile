# Build using g++
# CXX = g++
# MPICXX = mpicxx -cxx=g++ -DUSE_MPI
# CXXFLAGS := -std=c++11 -Wall -Wextra -O0
# CXXFLAGS += -g -g3 -ggdb -Wall -fno-elide-constructors

#MODE=debug
MODE=release

ifeq ($(MODE),debug)
	OPT = -O0
	DEBUG = -DTAPAS_DEBUG=1 -g -gdwarf-2
	LOG = -DTAPAS_LOG_LEVEL=1
else
	OPT = -O3
	DEBUG = -DTAPAS_DEBUG=0
	LOG = -DTAPAS_LOG_LEVEL=0
endif

CXX = clang++
MPICXX = mpicxx -cxx=clang++ -DUSE_MPI
CXXFLAGS := -std=c++11 -fcolor-diagnostics -Wall -Wextra
CXXFLAGS += -Wall -fno-elide-constructors

CXXFLAGS += -DS=$(S)

CXXFLAGS += -DTAPAS_BH # temporary cxxflag for Tapas Barnes-Hut implementation.

TAPAS_DIR = ../../cpp

all: bh bh_mpi

bh.o: bh.cc
	$(CXX) $(CXXFLAGS) $(DEBUG) $(LOG) $(OPT) -c $^ -I$(TAPAS_DIR)/include

bh: bh.o
	$(CXX) $^ -o $@

bh_mpi.o: bh.cc
	$(MPICXX) -o $@ $(CXXFLAGS) $(OPT) $(DEBUG) $(LOG) -c $^ -I$(TAPAS_DIR)/include

bh_mpi: bh_mpi.o
	$(MPICXX) $^ $(PROF) -o $@

clean:
	$(RM) -f *.o bh bh_mpi
