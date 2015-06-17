CXX = clang++
MPICXX = mpicxx -cxx=clang++ -DBH_USE_MPI
#CXX = clang++ -std=c++11 -fcolor-diagnostics 
#CXX = g++ -std=c++11

CXXFLAGS := -std=c++11 -fcolor-diagnostics -Wall -Wextra
CXXFLAGS += -g3 -Wall -fno-elide-constructors 

ifeq ($(NB),)
NB = 1024
endif
ifeq ($(S),)
S = 10
endif

#DEBUG := -DDUMP
LOG_FLAGS := -DTAPAS_LOG_LEVEL=1

CXX += -DNB=$(NB) -DS=$(S)

TAPAS_DIR = ../../cpp

all: bh bh_mpi

bh.o: bh.cc
	$(CXX) $(CXXFLAGS) $(DEBUG) $(LOG_FLAGS) -O2 -c $^ -Wall -I$(TAPAS_DIR)/include

bh: bh.o
	$(CXX) $^ -o $@

bh_mpi.o: bh.cc
	$(MPICXX) -o $@ $(CXXFLAGS) $(DEBUG) $(LOG_FLAGS) -O2 -c $^ -Wall -I$(TAPAS_DIR)/include

bh_mpi: bh_mpi.o
	$(MPICXX) $^ -o $@

clean:
	$(RM) -f *.o bh
