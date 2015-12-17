.PHONY : all clean
targets = test_set test_morton_key test_allgatherv test_alltoallv test_cpptricks
all: $(targets)

CXXFLAGS := -std=c++11 -Wall -Wextra -I../include -Wno-sign-compare -DUSE_MPI
MPICXX ?= mpicxx -cxx=$(CXX)

$(targets): % : %.cpp
	$(MPICXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(targets)
