CXX = clang++ -std=c++11 -fcolor-diagnostics -Wall -Wextra
#CXX = clang++ -std=c++11 -fcolor-diagnostics 
#CXX = g++ -std=c++11

CXX += -O0 -g3 -Wall -fno-elide-constructors

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

all: bh

bh.o: bh.cc
	$(CXX) $(DEBUG) $(LOG_FLAGS) -O2 -c $^ -Wall -I$(TAPAS_DIR)/include

bh: bh.o
	$(CXX) $^ -o $@

clean:
	$(RM) -f *.o bh
