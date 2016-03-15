/* vectormap_cuda.h -*- Coding: us-ascii-unix; -*- */
/* Copyright (C) 2015-2015 RIKEN AICS */

#ifndef TAPAS_VECTORMAP_CUDA_H_
#define TAPAS_VECTORMAP_CUDA_H_

/** @file vectormap_cuda.h @brief Direct part by CUDA.  See
    "vectormap_cpu.h". */

#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>

#include "tapas/vectormap_util.h"

#include <atomic>
#include <mutex>
#include <chrono>

namespace tapas {

/* Table of core counts on an SM by compute-capability. (There is
   likely no way to get the core count; See deviceQuery in CUDA
   samples). */

static struct TESLA_CORES {int sm; int n_cores;} tesla_cores[] = {
  {10, 8}, {11, 8}, {12, 8}, {13, 8},
  {20, 32}, {21, 48},
  {30, 192}, {32, 192}, {35, 192}, {37, 192},
  {50, 128},
};

static std::atomic<int> streamid (0);

/* The number of command streams. */

#define TAPAS_CUDA_MAX_NSTREAMS 128

/* GPU State of A Process.  It assumes use of a single GPU for each
   MPI process.  NSTREAMS is the number of command streams (There is
   likely no bound of the count). (32 maximum concurrent kernels on
   Kepler sm_35).  NCONNECTIONS is the number of physical command
   streams. (default=8 and maximum=32 on Kepler sm_35). */

struct TESLA {
  int gpuno;
  int sm;
  int n_sm;
  int n_cores;
  size_t scratchpad_size;
  size_t max_cta_size;
  int kernel_max_cta_size;
  /* Tapas options */
  int cta_size;
  int n_streams;
  int n_connections;
#ifdef __CUDACC__
  cudaStream_t streams[TAPAS_CUDA_MAX_NSTREAMS];
#endif
};

#define TAPAS_CEILING(X,Y) (((X) + (Y) - 1) / (Y))
#define TAPAS_FLOOR(X,Y) ((X) / (Y))

namespace {
inline void vectormap_check_error(const char *msg, const char *file, const int line) {
  cudaError_t ce = cudaGetLastError();
  if (ce != cudaSuccess) {
    fprintf(stderr,
            "%s:%i (%s): CUDA ERROR (%d): %s.\n",
            file, line, msg, (int)ce, cudaGetErrorString(ce));
    //cudaError_t ce1 = cudaDeviceReset();
    //assert(ce1 == cudaSuccess);
    assert(ce == cudaSuccess);
  }
}
} // anon namespace

#if 0
template <class Funct, typename BV, class... Args>
__global__
void vectormap_cuda_kernel1(BV* v0, size_t n0,
                            Funct f, Args... args) {
  int index = (blockDim.x * blockIdx.x + threadIdx.x);
  if (index < n0) {
    BV p0 = v0[index];
    f(p0, args...);
  }
}
#endif

/* (Two argument mapping (each pair)) */

/* Accumulates partial acceleration for the 1st vector.  Blocking
   size of the 2nd vector is passed as TILESIZE. */

template <class Funct, class BV, class BA, class... Args>
__global__
void vectormap_cuda_plain_kernel2(BV* v0, BV* v1, BA* a0,
                                  size_t n0, size_t n1, int tilesize,
                                  Funct f, Args... args) {
  assert(tilesize <= blockDim.x);
  int index = (blockDim.x * blockIdx.x + threadIdx.x);
  extern __shared__ BV scratchpad[];
  int ntiles = TAPAS_CEILING(n1, tilesize);
  BV* p0 = ((index < n0) ? &v0[index] : &v0[0]);
  BA q0 = ((index < n0) ? a0[index] : a0[0]);
  for (int t = 0; t < ntiles; t++) {
    if ((tilesize * t + threadIdx.x) < n1 && threadIdx.x < tilesize) {
      scratchpad[threadIdx.x] = v1[tilesize * t + threadIdx.x];
    }
    __syncthreads();

    if (index < n0) {
      unsigned int jlim = min(tilesize, (int)(n1 - tilesize * t));
      /*AHO*/ //#pragma unroll 128
      for (unsigned int j = 0; j < jlim; j++) {
        BV* p1 = &scratchpad[j];
        if (!(v0 == v1 && index == (tilesize * t + j))) {
          f(p0, p1, q0, args...);
        }
      }
    }
    __syncthreads();
  }
  if (index < n0) {
    a0[index] = q0;
  }
}

/* (Atomic-add code from cuda-c-programming-guide). */

__device__
static double atomicAdd(double* address, double val) {
  // Should we use uint64_t ?
  static_assert(sizeof(unsigned long long int) == sizeof(double),   "sizeof(unsigned long long int) == sizeof(double)");
  static_assert(sizeof(unsigned long long int) == sizeof(uint64_t), "sizeof(unsigned long long int) == sizeof(uint64_t)");
  
  unsigned long long int* address1 = (unsigned long long int*)address;
  unsigned long long int chk;
  unsigned long long int old;
  chk = *address1;
  do {
    old = chk;
    chk = atomicCAS(address1, old,
                    __double_as_longlong(val + __longlong_as_double(old)));
  } while (old != chk);
  return __longlong_as_double(old);
}

__device__
static float atomicAdd(float* address, float val) {
  // Should we use uint32_t ?
  static_assert(sizeof(int) == sizeof(float), "sizeof(int) == sizeof(float)");
  static_assert(sizeof(uint32_t) == sizeof(float), "sizeof(int) == sizeof(float)");
  
  int* address1 = (int*)address;
  int chk;
  int old;
  chk = *address1;
  do {
    old = chk;
    chk = atomicCAS(address1, old,
                    __float_as_int(val + __int_as_float(old)));
  } while (old != chk);
  return __int_as_float(old);
}

template <class Funct, class BV, class BA,
          template <class T> class CELLDATA, class... Args>
__global__
void vectormap_cuda_pack_kernel2(CELLDATA<BV>* v, CELLDATA<BA>* a,
                                 size_t nc,
                                 int rsize, BV* rdata, int tilesize,
                                 Funct f, Args... args) {
  static_assert(std::is_same<BA, kvec4>::value, "attribute type=kvec4");

  assert(tilesize <= blockDim.x);
  int index = (blockDim.x * blockIdx.x + threadIdx.x);
  extern __shared__ BV scratchpad[];

  int cell = -1;
  int item = 0;
  int base = 0;
  for (int c = 0; c < nc; c++) {
    if (base <= index && index < base + v[c].size) {
      assert(cell == -1);
      cell = c;
      item = (index - base);
    }
    base += (TAPAS_CEILING(v[c].size, 32) * 32);
  }

  int ntiles = TAPAS_CEILING(rsize, tilesize);
  BV &p0 = (cell != -1) ? v[cell].data[item] : v[0].data[0];
  BA q0 = {0.0f, 0.0f, 0.0f, 0.0f}; // bzero?

  for (int t = 0; t < ntiles; t++) {
    if ((tilesize * t + threadIdx.x) < rsize && threadIdx.x < tilesize) {
      scratchpad[threadIdx.x] = rdata[tilesize * t + threadIdx.x];
    }
    __syncthreads();

    if (cell != -1) {
      unsigned int jlim = min(tilesize, (int)(rsize - tilesize * t));
#pragma unroll 128
      for (unsigned int j = 0; j < jlim; j++) {
        BV &p1 = scratchpad[j];
        f(&p0, &p1, q0, args...); // q0 -> biattr
      }
    }
    __syncthreads();
  }

  if (cell != -1) {
    // Really necessary?
    assert(item < a[cell].size);
    BA &a0 = a[cell].data[item]; // FIXME: Dependency to ExaFMM !!!
    atomicAdd(&(a0[0]), q0[0]);
    atomicAdd(&(a0[1]), q0[1]);
    atomicAdd(&(a0[2]), q0[2]);
    atomicAdd(&(a0[3]), q0[3]);
  }
}

template<class T0, class T1, class T2>
struct cellcompare_r {
  bool operator() (const std::tuple<T0, T1, T2> &i,
                   const std::tuple<T0, T1, T2> &j) {
    return ((std::get<2>(i).data) < (std::get<2>(j).data));
  }
};



template<int _DIM, class _FP, class _BT, class _BT_ATTR>
struct Vectormap_CUDA_Simple {

  using Body = _BT;
  using BodyAttr = _BT_ATTR;

  TESLA tesla_dev_;

  /**
   * \brief Memory allocator for the unified memory.  
   * It will replace the
   * vector allocators.  (N.B. Its name should be generic because it
   * is used in CPUs also.)
   */
  template <typename T>
  struct um_allocator : public std::allocator<T> {
   public:
    /*typedef T* pointer;*/
    /*typedef const T* const_pointer;*/
    /*typedef T value_type;*/
    template <class U> struct rebind {typedef um_allocator<U> other;};

    T* allocate(size_t n, const void* hint = 0) {
      T* p;
      CUDA_SAFE_CALL(cudaMallocManaged(&p, (sizeof(T) * n), cudaMemAttachGlobal));
      assert(p != nullptr);
      fprintf(stderr, ";; cudaMallocManaged() p=%p n=%zd\n", p, n); fflush(0);
      return p;
    }

    void deallocate(T* p, size_t n) {
      CUDA_SAFE_CALL(cudaFree(p));
      fprintf(stderr, ";; cudaFree() p=%p n=%zd\n", p, n); fflush(0);
    }

    explicit um_allocator() throw() : std::allocator<T>() {}

    /*explicit*/ um_allocator(const um_allocator<T> &a) throw()
      : std::allocator<T>(a) {}

    template <class U> explicit
    um_allocator(const um_allocator<U> &a) throw()
      : std::allocator<T>(a) {}

    ~um_allocator() throw() {}
  };

  void setup(int cta, int nstreams) {
    assert(nstreams <= TAPAS_CUDA_MAX_NSTREAMS);

    tesla_dev_.cta_size = cta;
    tesla_dev_.n_streams = nstreams;

#ifdef USE_MPI

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rankofnode, rankinnode, nprocsinnode;
    rank_in_node(MPI_COMM_WORLD, rankofnode, rankinnode, nprocsinnode);

#else /* USE_MPI */

    int rank = 0;
    int rankinnode = 0;
    int nprocsinnode = 1;

#endif /* USE_MPI */

    int ngpus;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&ngpus));
    if (ngpus < nprocsinnode) {
      fprintf(stderr, "More ranks than GPUs on a node\n");
      assert(ngpus >= nprocsinnode);
    }

    tesla_dev_.gpuno = rankinnode;
    cudaDeviceProp prop;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, tesla_dev_.gpuno));
    CUDA_SAFE_CALL(cudaSetDevice(tesla_dev_.gpuno));

    printf(";; Rank#%d uses GPU#%d\n", rank, tesla_dev_.gpuno);

    assert(prop.unifiedAddressing);

    tesla_dev_.sm = (prop.major * 10 + prop.minor);
    tesla_dev_.n_sm = prop.multiProcessorCount;

    tesla_dev_.n_cores = 0;
    for (struct TESLA_CORES &i : tesla_cores) {
      if (i.sm == tesla_dev_.sm) {
        tesla_dev_.n_cores = i.n_cores;
        break;
      }
    }
    assert(tesla_dev_.n_cores != 0);

    tesla_dev_.scratchpad_size = prop.sharedMemPerBlock;
    tesla_dev_.max_cta_size = prop.maxThreadsPerBlock;
    assert(prop.maxThreadsPerMultiProcessor >= prop.maxThreadsPerBlock * 2);

    for (int i = 0; i < tesla_dev_.n_streams; i++) {
      CUDA_SAFE_CALL(cudaStreamCreate(&tesla_dev_.streams[i]));
    }
  }
  
  void release() {
    for (int i = 0; i < tesla_dev_.n_streams; i++) {
      CUDA_SAFE_CALL(cudaStreamDestroy(tesla_dev_.streams[i]));
    }
  }

  void start() {}

  void finish() {
    vectormap_check_error("vectormap_end", __FILE__, __LINE__);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
  }

  /* (One argument mapping) */

  /* NOTE IT RUNS ON CPUs.  The kernel "tapas_kernel::L2P()" is not
     coded to be run on GPUs, since it accesses the cell. */

  template <class Funct, class Cell, class... Args>
  void vector_map1(Funct f, BodyIterator<Cell> iter,
                          Args... args) {
    int sz = iter.size();
    for (int i = 0; i < sz; i++) {
      f(*(iter + i), args...);
    }
  }

#if 0
  template <class Funct, class Cell, class... Args>
  static void vector_map1(Funct f,
                          BodyIterator<Cell> b0,
                          Args... args) {
    static std::mutex mutex0;
    static struct cudaFuncAttributes tesla_attr0;
    if (tesla_attr0.binaryVersion == 0) {
      mutex0.lock();
      CUDA_SAFE_CALL(cudaFuncGetAttributes(
          &tesla_attr0,
          &vectormap_cuda_kernel1<Funct, Body, Args...>));
      mutex0.unlock();
    }
    assert(tesla_attr0.binaryVersion != 0);

    size_t n0 = b0.size();
    int n0up = (TAPAS_CEILING(n0, 256) * 256);
    int ctasize = std::min(n0up, tesla_attr0.maxThreadsPerBlock);
    size_t nblocks = TAPAS_CEILING(n0, ctasize);

    streamid++;
    int s = (streamid % tesla_dev_.n_streams);
    vectormap_cuda_kernel1<<<nblocks, ctasize, 0, tesla_dev_.streams[s]>>>
      (b0, n0, f, args...);
  }
#endif

  /* (Two argument mapping) */

  /* Implements a map on a GPU.  It extracts vectors of bodies.  It
     uses a fixed command stream to serialize processing on each cell.
     A call to cudaDeviceSynchronize() is needed on the caller of
     Tapas-map.  The CTA size is the count in the first cell rounded
     up to multiples of 256.  The tile size is the count in the first
     cell rounded down to multiples of 64 (tile size is the count of
     preloading of the second cells). */

  template <class Funct, class Cell, class... Args>
  void vectormap_cuda_plain(Funct f, Cell &c0, Cell &c1, Args... args) {
    using BV = Body;
    using BA = BodyAttr;

    static std::mutex mutex1;
    static struct cudaFuncAttributes tesla_attr1;
    if (tesla_attr1.binaryVersion == 0) {
      mutex1.lock();
      CUDA_SAFE_CALL(cudaFuncGetAttributes(
          &tesla_attr1,
          &vectormap_cuda_plain_kernel2<Funct, BV, BA, Args...>));
      mutex1.unlock();
    }
    assert(tesla_attr1.binaryVersion != 0);

    assert(c0.IsLeaf() && c1.IsLeaf());
    /* (Cast to drop const, below). */
    BV* v0 = (BV*)&(c0.body(0));
    BV* v1 = (BV*)&(c1.body(0));
    BA* a0 = (BA*)&(c0.body_attr(0));
    size_t n0 = c0.nb();
    size_t n1 = c1.nb();
    assert(n0 != 0 && n1 != 0);

    /*bool am = AllowMutual<T1_Iter, T2_Iter>::value(b0, b1);*/
    /*int n0up = (TAPAS_CEILING(n0, 256) * 256);*/
    /*int n0up = (TAPAS_CEILING(n0, 32) * 32);*/
    int cta0 = (TAPAS_CEILING(tesla_dev_.cta_size, 32) * 32);
    int ctasize = std::min(cta0, tesla_attr1.maxThreadsPerBlock);
    assert(ctasize == tesla_dev_.cta_size);

    int tile0 = (tesla_dev_.scratchpad_size / sizeof(Body));
    int tile1 = (TAPAS_FLOOR(tile0, 32) * 32);
    int tilesize = std::min(ctasize, tile1);
    assert(tilesize > 0);

    int scratchpadsize = (sizeof(Body) * tilesize);
    size_t nblocks = TAPAS_CEILING(n0, ctasize);

#if 0 /*AHO*/
    fprintf(stderr, "launch arrays=(%p/%ld, %p/%ld, %p/%ld) blks=%ld cta=%d\n",
            v0, n0, v1, n1, a0, n0, nblocks, ctasize);
    fflush(0);
#endif

    int s = (((unsigned long)&c0 >> 4) % tesla_dev_.n_streams);
    vectormap_cuda_plain_kernel2<<<nblocks, ctasize, scratchpadsize,
      tesla_dev_.streams[s]>>>
      (v0, v1, a0, n0, n1, tilesize, f, args...);
  }

  /** 
   * \fn Vectormap_CUDA_Simple::vector_map2
   * \brief Calls a function FN given by the user on each data pair in the
   *        cells.  f takes arguments of Body&, Body&,
   *        BodyAttr&, and extra call arguments. 
   */
  template <class Funct, class Cell, class...Args>
  void vector_map2(Funct f, ProductIterator<BodyIterator<Cell>> prod,
                   Args... args) {
    printf("vector_map2X\n"); fflush(0);
    
    typedef BodyIterator<Cell> Iter;
    const Cell &c0 = prod.first().cell();
    const Cell &c1 = prod.second().cell();
    if (c0 == c1) {
      vectormap_cuda_plain(f, c0, c1, args...);
    } else {
      vectormap_cuda_plain(f, c0, c1, args...);
      //vectormap_cuda_plain(f, c1, c0, args...); // mutual is not supported 
    }
  }

  inline TESLA& tesla_dev() { return tesla_dev_; } // used in the child class
};

template <class T>
struct Cell_Data {
  int size;
  T* data;
};

// Launches a kernel on Tesla.
// Used by Vectormap_CUDA_Pakced and Applier.
template <class Caller, class Funct, class... Args>
void invoke(Caller *caller, int start, int nc, Cell_Data<Body> &r,
            int tilesize, size_t nblocks, int ctasize, int scratchpadsize,
            Funct f, Args... args) {
  using BV = typename Caller::Body;
  using BA = typename Caller::BodyAttr;

  TESLA &tesla_dev = caller->tesla_dev();
  
  /*AHO*/
  if (0) {
    printf("kernel(nblocks=%ld ctasize=%d scratchpadsize=%d tilesize=%d\n",
           nblocks, ctasize, scratchpadsize, tilesize);
    printf("invoke(start=%d ncells=%d)\n", start, nc);

    for (int i = 0; 0 && i < nc; i++) {
      Cell_Data<BV> &lc = std::get<0>(caller->cellpairs_[start + i]);
      Cell_Data<BA> &ac = std::get<1>(caller->cellpairs_[start + i]);
      Cell_Data<BV> &rc = std::get<2>(caller->cellpairs_[start + i]);
      assert(rc.data == r.data);
      assert(ac.size == lc.size);
      printf("pair(celll=%p[%d] cellr=%p[%d])\n",
             lc.data, lc.size, rc.data, rc.size);
    }
    fflush(0);
  }

  streamid++;
  int s = (streamid % tesla_dev.n_streams);
  vectormap_cuda_pack_kernel2<<<nblocks, ctasize, scratchpadsize,
      tesla_dev.streams[s]>>>
      (&(caller->dvcells_[start]), &(caller->dacells_[start]), nc, r.size, r.data,
       tilesize, f, args...);
}

// Utility routine to support delayed dispatch of function template with variadic parameters.
template<int ...>
struct seq { };

template<int N, int ...S>
struct gens : gens<N-1, N-1, S...> { };

template<int...S>
struct gens<0, S...> {
  typedef seq<S...> type;
};

// Abstract base class of Applier.
// Subclasses take a particular function type (ex. P2P) and variadic arguments.
template<class Vectormap>
class AbstractApplier {
 public:
  virtual void apply(Vectormap *vm) = 0;
};

// When tapas::Map <Body x Body>
template<class Vectormap, class Funct, class...Args>
class Applier : public AbstractApplier<Vectormap> {
  Funct f_;
  std::tuple<Args...> args_;
  std::mutex mutex_;
  cudaFuncAttributes func_attrs_;
  
  using ParamIdxSeq = typename gens<sizeof...(Args)>::type; // used to hold args... for invoke

 public:

  using Body = typename Vectormap::Body;
  using BodyAttr = typename Vectormap::BodyAttr;

  // Call ::invoke() function with args... 
  template<int ...ParamIdx>
  inline void invoke2(Vectormap *caller, int start, int nc, Cell_Data<Body> &r,
                      int tilesize, size_t nblocks, int ctasize, int scratchpadsize,
                      seq<ParamIdx...>) {
    invoke(caller, start, nc, r, tilesize, nblocks, ctasize, scratchpadsize, f_, std::get<ParamIdx>(args_)...);
  }

  // ctor. not thread safe.
  Applier(Funct f, Args... args) : f_(f), args_(args...), func_attrs_() {
    using BV = Body;
    using BA = BodyAttr;
    if (func_attrs_.binaryVersion == 0) {
      CUDA_SAFE_CALL(cudaFuncGetAttributes(
          &func_attrs_,
          &vectormap_cuda_pack_kernel2<Funct, BV, BA, Cell_Data, Args...>));

#ifdef TAPAS_DEBUG
      fprintf(stderr,
              ";; vectormap_cuda_pack_kernel2:"
              " binaryVersion=%d, cacheModeCA=%d, constSizeBytes=%zd,"
              " localSizeBytes=%zd, maxThreadsPerBlock=%d, numRegs=%d,"
              " ptxVersion=%d, sharedSizeBytes=%zd\n",
              func_attrs_.binaryVersion,      func_attrs_.cacheModeCA,
              func_attrs_.constSizeBytes,     func_attrs_.localSizeBytes,
              func_attrs_.maxThreadsPerBlock, func_attrs_.numRegs,
              func_attrs_.ptxVersion,         func_attrs_.sharedSizeBytes);
#endif
    }

    TAPAS_ASSERT(func_attrs_.binaryVersion != 0);
  }

  /**
   * @brief Invoke CUDA kernel with the function (of type Funct) to the cellpairs list.
   */
  virtual void apply(Vectormap *vm) override {
    using BV = Body;
    using BA = BodyAttr;
    using namespace std::chrono;

    using CellTuple = std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>;

    auto t0 = high_resolution_clock::now();

    TESLA &tesla_dev = vm->tesla_dev();

    if (vm->cellpairs_.size() == 0) {
      return;
    }

    printf(";; pairs=%ld\n", vm->cellpairs_.size());

    // cta = cooperative thread array = thread block
    int cta0 = (TAPAS_CEILING(tesla_dev.cta_size, 32) * 32);
    int ctasize = std::min(cta0, func_attrs_.maxThreadsPerBlock);
    assert(ctasize == tesla_dev.cta_size);

    int tile0 = (tesla_dev.scratchpad_size / sizeof(Body));
    int tile1 = (TAPAS_FLOOR(tile0, 32) * 32);
    int tilesize = std::min(ctasize, tile1);
    assert(tilesize > 0);

    int scratchpadsize = (sizeof(Body) * tilesize);
    size_t nblocks = TAPAS_CEILING(Vectormap::N0, ctasize);

    // Re-use pre-allocated memory region, or re-allocate if neceessary
    if (vm->npairs_ < vm->cellpairs_.size()) {
      CUDA_SAFE_CALL( cudaFree(vm->dvcells_) );
      CUDA_SAFE_CALL( cudaFree(vm->dacells_) );
      CUDA_SAFE_CALL( cudaFree(vm->hvcells_) );
      CUDA_SAFE_CALL( cudaFree(vm->hacells_) );

      // dvcells : Device bodies' Values memory
      // dacells : Device bodies' Attrs memory
      // hvcells : Host boddies' Values memory
      // hacells : Host bodies' Attrs  memory
      vm->npairs_ = vm->cellpairs_.size();
      CUDA_SAFE_CALL( cudaMalloc(&vm->dvcells_, (sizeof(Cell_Data<BV>) * vm->npairs_)) );
      CUDA_SAFE_CALL( cudaMalloc(&vm->dacells_, (sizeof(Cell_Data<BA>) * vm->npairs_)) );
      CUDA_SAFE_CALL( cudaMallocHost(&vm->hvcells_, (sizeof(Cell_Data<BV>) * vm->npairs_)) );
      CUDA_SAFE_CALL( cudaMallocHost(&vm->hacells_, (sizeof(Cell_Data<BA>) * vm->npairs_)) );
    }

    auto t1 = high_resolution_clock::now();
    
    auto comp = cellcompare_r<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>();

    std::sort(vm->cellpairs_.begin(), vm->cellpairs_.end(), comp);
    for (size_t i = 0; i < vm->npairs_; i++) {
      std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>> &c = vm->cellpairs_[i];
      vm->hvcells_[i] = std::get<0>(c);
    }
    CUDA_SAFE_CALL(cudaMemcpy(vm->dvcells_, vm->hvcells_, (sizeof(Cell_Data<BV>) * vm->npairs_),
                              cudaMemcpyHostToDevice));
    
    for (size_t i = 0; i < vm->npairs_; i++) {
      std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>> &c = vm->cellpairs_[i];
      vm->hacells_[i] = std::get<1>(c);
    }
    CUDA_SAFE_CALL(cudaMemcpy(vm->dacells_, vm->hacells_, (sizeof(Cell_Data<BA>) * vm->npairs_),
                              cudaMemcpyHostToDevice));
    
    auto t2 = high_resolution_clock::now();
    
    Cell_Data<BV> xr = std::get<2>(vm->cellpairs_[0]);
    int xncells = 0;
    int xndata = 0;
    
    for (size_t i = 0; i < vm->npairs_; i++) {
      CellTuple &c = vm->cellpairs_[i];
      Cell_Data<BV> &r = std::get<2>(c);
      if (xr.data != r.data) {
        assert(i != 0 && xncells > 0);
        invoke2(vm, (i - xncells), xncells, xr,
                tilesize, nblocks, ctasize, scratchpadsize,
                ParamIdxSeq());
        xncells = 0;
        xndata = 0;
        xr = r;
      }
      Cell_Data<BV> &l = std::get<0>(c);
      size_t nb = TAPAS_CEILING((xndata + l.size), ctasize);
      //std::cerr << "nb = " << nb << ", nblocks = " << nblocks << std::endl;
      if (nb > nblocks) {
        //std::cerr << "i = " << i << ", xncells = " << xncells << std::endl;
        assert(i != 0 && xncells > 0);
        invoke2(vm, (i - xncells), xncells, xr,
                tilesize, nblocks, ctasize, scratchpadsize,
                ParamIdxSeq());
        xncells = 0;
        xndata = 0;
        xr = r;
      }
      xncells++;
      xndata += (TAPAS_CEILING(l.size, 32) * 32);
    }
    assert(xncells > 0);
    invoke2(vm, (vm->npairs_ - xncells), xncells, xr,
            tilesize, nblocks, ctasize, scratchpadsize,
            ParamIdxSeq());
    
    auto t3 = high_resolution_clock::now();

    // Report time (In a ad-hoc way using std::cout. Needs refactoring)
    double time_total = duration_cast<microseconds>(t3-t0).count() * 1e-6;
    double time_mcopy = duration_cast<microseconds>(t2-t1).count() * 1e-6;
    double time_launch = duration_cast<microseconds>(t3-t2).count() * 1e-6;
    double time_other = time_total - time_mcopy - time_launch;
  
    std::cout << "CUDA map2 memcopy    : " << std::scientific << time_mcopy  << " s" << std::endl;
    std::cout << "CUDA map2 launch     : " << std::scientific << time_launch << " s" << std::endl;
    std::cout << "CUDA map2 other      : " << std::scientific << time_other  << " s" << std::endl;
    std::cout << "CUDA map2 total      : " << std::scientific << time_total  << " s" << std::endl;
  }
};

template<int _DIM, class _FP, class _BT, class _BT_ATTR>
struct Vectormap_CUDA_Packed
    : public Vectormap_CUDA_Simple<_DIM, _FP, _BT, _BT_ATTR> {
  using BV = _BT;
  using BA = _BT_ATTR;

  using Body = _BT;
  using BodyAttr = _BT_ATTR;
  
  using CellPair = std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>;
  using Vectormap = Vectormap_CUDA_Packed<_DIM, _FP, _BT, _BT_ATTR>; // self

  // Pairs of bodies to which the user function is applied
  std::vector<CellPair> cellpairs_;
  std::mutex pairs_mutex_;
  size_t npairs_;

  Cell_Data<BV>* dvcells_;
  Cell_Data<BV>* hvcells_;
  Cell_Data<BA>* dacells_;
  Cell_Data<BA>* hacells_;

  // funct_id_ is used to check if the same Funct is used for all cell pairs.
  // In the current implementation, it is assumed that a single function and
  // the same optional arguments (= Args...) are used to all cell pairs.
  std::mutex applier_mutex_;
  intptr_t funct_id_;
  AbstractApplier<Vectormap> *applier_;

  cudaFuncAttributes func_attrs_;

  double time_device_call_;
  
  void start() {
    //printf(";; start\n"); fflush(0);
    cellpairs_.clear();
  }

  /**
   * @brief ctor.
   * not thread safe
   */
  Vectormap_CUDA_Packed()
      : npairs_(0)
      , dvcells_(nullptr)
      , hvcells_(nullptr)
      , dacells_(nullptr)
      , hacells_(nullptr)
      , applier_mutex_()
      , funct_id_(0)
      , applier_(nullptr)
      , time_device_call_(0)
  { }

  /* (Two argument mapping with left packing.) */

  template <class Cell, class Funct, class... Args>
  void map2(Funct f, ProductIterator<BodyIterator<Cell>> prod, Args... args) {
    using BV = Body;
    using BA = BodyAttr;

    static_assert(std::is_same<typename Cell::Body, Body>::value, "inconsistent Cell and Body types");
    static_assert(std::is_same<typename Cell::BodyAttr, BodyAttr>::value, "inconsistent Cell and BodyAttr types");

    const Cell &c0 = prod.first().cell();
    const Cell &c1 = prod.second().cell();
    assert(c0.IsLeaf() && c1.IsLeaf());

    if (c0.nb() == 0 || c1.nb() == 0) return;

    // Create Applier with Funct and Args...
    if (applier_ == nullptr) {
      applier_mutex_.lock();
      if (applier_ == nullptr) {
        applier_ = new Applier<Vectormap, Funct, Args...>(f, args...);
        funct_id_ = Type2Int<Funct>::value();

        // Memo [Jan 18, 2016]
        // func_id_ is not used as of now. This check integer is for when there are multiple kernels
        // for bodies x bodies product map.
        // An interaction list is created for each function (stored in a unordered_map of which keys are integers).
      }
      applier_mutex_.unlock();
    }
    
    TAPAS_ASSERT(funct_id_ == Type2Int<Funct>::value());

    /* (Cast to drop const, below). */
    Cell_Data<BV> d0;
    Cell_Data<BV> d1;
    Cell_Data<BA> a0;
    //Cell_Data<BA> a1;
    d0.size = c0.nb();
    d0.data = (BV*)&(c0.body(0));
    a0.size = c0.nb();
    a0.data = (BA*)&(c0.body_attr(0));
    d1.size = c1.nb();
    d1.data = (BV*)&(c1.body(0));
    //a1.size = c1.nb();
    //a1.data = (BA*)&(c1.body_attr(0));
    
    if (c0 == c1) {
      pairs_mutex_.lock();
      cellpairs_.push_back(std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>(d0, a0, d1));
      pairs_mutex_.unlock();
    } else {
      pairs_mutex_.lock();
      cellpairs_.push_back(std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>(d0, a0, d1));
      //cellpairs_.push_back(std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>(d1, a1, d0)); // mutual is not supported in CUDA version
      pairs_mutex_.unlock();
    }
  }
  
  /* Limit of the number of threads in grids. */

  static const constexpr int N0 = (16 * 1024);

  /* Starts launching a kernel on collected cells. */
  
  void on_collected() {
    auto t1 = std::chrono::high_resolution_clock::now();
    
    applier_->apply(this);

    auto t2 = std::chrono::high_resolution_clock::now();
    
    time_device_call_ = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() * 1e-6;
  }
  
  void finish() {
    //printf(";; Vectormap_CUDA_Packed::finish\n"); fflush(0);
    on_collected();
    vectormap_check_error("Vectormap_CUDA_Packed::end", __FILE__, __LINE__);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    if (applier_ != nullptr) {
      applier_mutex_.lock();
      if (applier_ != nullptr) {
        delete applier_;
        applier_ = nullptr;
        funct_id_ = 0;
      }
      applier_mutex_.unlock();
    }
  }
};

}

#endif /*__CUDACC__*/

#endif /*TAPAS_VECTORMAP_CUDA_H_*/
