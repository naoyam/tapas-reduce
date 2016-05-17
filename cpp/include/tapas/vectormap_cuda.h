
/* vectormap_cuda.h -*- Coding: us-ascii-unix; -*- */
/* Copyright (C) 2015-2015 RIKEN AICS */

#include <type_traits>

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

/**
 * \breif Atomic-add code from cuda-c-programming-guide (double precision version)
 */
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

/**
 * \breif Atomic-add code from cuda-c-programming-guide (double precision version)
 */
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

template<class T0, class T1, class T2>
struct cellcompare_r {
  bool operator() (const std::tuple<T0, T1, T2> &i,
                   const std::tuple<T0, T1, T2> &j) {
    return ((std::get<2>(i).data) < (std::get<2>(j).data));
  }
};

} // anon namespace

/**
 * \brief Single argument mapping over bodies on GPU
 */
template <class CA, class V3, class BT, class BT_ATTR, class Funct, class... Args>
__global__
void vectormap_cuda_plain_kernel1(const CA c_attr, const V3 c_center,
                                  const BT* b, BT_ATTR* b_attr,
                                  size_t sz, Funct f, Args... args) {
  int index = (blockDim.x * blockIdx.x + threadIdx.x);
  if (index < sz) {
    f(c_attr, c_center, (b + index), (b_attr + index), args...);
  }
}

template <class Funct, class BT, class BT_ATTR, class CELL_ATTR, class VEC,
          template <class T> class CELLDATA, class... Args>
__global__
void vectormap_cuda_pack_kernel1(int nmapdata, size_t nbodies,
                                 CELLDATA<BT>* body_list,
                                 CELLDATA<BT_ATTR>* attr_list,
                                 CELL_ATTR* cell_attrs,
                                 VEC* cell_centers,
                                 Funct f, Args... args) {
  static_assert(std::is_same<BT_ATTR, kvec4>::value, "attribute type=kvec4");

  int index = (blockDim.x * blockIdx.x + threadIdx.x);
  if (index < nbodies) {
    int bodycount = 0;
    int nth = -1;
    for (size_t i = 0; i < nmapdata; i++) {
      if (index < (bodycount + body_list[i].size)) {
        nth = i;
        break;
      } else {
        bodycount += body_list[i].size;
      }
    }
    assert(nth != -1);
    int nthbody = index - bodycount;
    f(cell_attrs[nth], cell_centers[nth],
      (body_list[nth].data + nthbody), (attr_list[nth].data + nthbody),
      args...);
  }
}

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


template <class Funct, class BT, class BT_ATTR,
          template <class T> class CELLDATA, class... Args>
__global__
void vectormap_cuda_pack_kernel2(CELLDATA<BT>* v, CELLDATA<BT_ATTR>* a,
                                 size_t nc,
                                 int rsize, BT* rdata, int tilesize,
                                 Funct f, Args... args) {
  // CELLDATA = Mirror_Data
  // nc= #cells
  static_assert(std::is_same<BT_ATTR, kvec4>::value, "attribute type=kvec4");

  assert(tilesize <= blockDim.x);
  int index = (blockDim.x * blockIdx.x + threadIdx.x);
  extern __shared__ BT scratchpad[];

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
  BT &p0 = (cell != -1) ? v[cell].data[item] : v[0].data[0]; // body value
  BT_ATTR q0 = {0.0f, 0.0f, 0.0f, 0.0f}; // bzero?
  BT_ATTR q1 = {0.0f, 0.0f, 0.0f, 0.0f}; // bzero?
  
  for (int t = 0; t < ntiles; t++) {
    // load body data in the tile t to the shared memory
    if ((tilesize * t + threadIdx.x) < rsize && threadIdx.x < tilesize) {
      scratchpad[threadIdx.x] = rdata[tilesize * t + threadIdx.x];
    }
    __syncthreads();
    
    if (cell != -1) {
      unsigned int jlim = min(tilesize, (int)(rsize - tilesize * t));
#pragma unroll 128
      for (unsigned int j = 0; j < jlim; j++) {
        BT &p1 = scratchpad[j];
        f(p0, q0, p1, q1, args...); // q0 -> biattr
      }
    }
    __syncthreads();
  }

  if (cell != -1) {
    assert(item < a[cell].size);
    BT_ATTR &a0 = a[cell].data[item];
    atomicAdd(&(a0[0]), q0[0]);
    atomicAdd(&(a0[1]), q0[1]);
    atomicAdd(&(a0[2]), q0[2]);
    atomicAdd(&(a0[3]), q0[3]);
  }
}

template<int _DIM, typename _FP, typename _BT, typename _BT_ATTR, typename _CELL_ATTR>
struct Vectormap_CUDA_Base {
  using Body = _BT;
  using BodyAttr = _BT_ATTR;
  using CellAttr = _CELL_ATTR;

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
      //CUDA_SAFE_CALL(cudaMallocManaged(&p, (sizeof(T) * n), cudaMemAttachHost));
      assert(p != nullptr);
      // fprintf(stderr, ";; cudaMallocManaged() p=%p n=%zd sizeof(T)=%zd size=%zd\n", p, n, sizeof(T), n * sizeof(T)); fflush(0);
      return p;
    }

    void deallocate(T* p, size_t n) {
      CUDA_SAFE_CALL(cudaFree(p));
      //fprintf(stderr, ";; cudaFree() p=%p n=%zd\n", p, n); fflush(0);
    }

    explicit um_allocator() throw() : std::allocator<T>() {}

    /*explicit*/ um_allocator(const um_allocator<T> &a) throw()
      : std::allocator<T>(a) {}

    template <class U> explicit
    um_allocator(const um_allocator<U> &a) throw()
      : std::allocator<T>(a) {}

    ~um_allocator() throw() {}
  }; // end of class um_allocator

  /**
   * CUDA GPU device information
   */
  TESLA tesla_dev_;
  inline TESLA& tesla_dev() { return tesla_dev_; }

  /**
   * \brief Setup CUDA devices: allocate 1 GPU per process (considering multiple processes per node)
   */
  void Setup(int cta, int nstreams) {
    assert(nstreams <= TAPAS_CUDA_MAX_NSTREAMS);

    tesla_dev_.cta_size = cta;
    tesla_dev_.n_streams = nstreams;

#ifdef USE_MPI

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rankofnode, rankinnode, nprocsinnode;
    rank_in_node(MPI_COMM_WORLD, rankofnode, rankinnode, nprocsinnode);
    //printf("rankofnode=%d, rankinnode=%, nprocsinnode=%d\n", rankofnode, rankinnode, nprocsinnode);

#else /* #ifdef USE_MPI */

    int rank = 0;
    int rankinnode = 0;
    int nprocsinnode = 1;

#endif /* USE_MPI */

    SetGPU();

    int ngpus;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&ngpus));
#if 0
    if (ngpus < nprocsinnode) {
      fprintf(stderr, "More ranks than GPUs on a node  ngpus = %d, nprocsinnode = %d\n", ngpus, nprocsinnode);
      assert(ngpus >= nprocsinnode);
    }
#endif

    // Since we assume CUDA_VISIBLE_DEVICES is properly set by SetGPU() function or by the user manually,
    // Each process should find 1 GPU.
    assert(ngpus == 1);

    tesla_dev_.gpuno = 0; // Fixed. Always use the first GPU  (see above).
    //tesla_dev_.gpuno = rankinnode;
    cudaDeviceProp prop;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, tesla_dev_.gpuno));
    CUDA_SAFE_CALL(cudaSetDevice(tesla_dev_.gpuno));
    
    //printf(";; Rank#%d uses GPU#%d\n", rank, tesla_dev_.gpuno); // always GPU#0

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

  /**
   * \brief Release the CUDA device
   */
  void Release() {
    for (int i = 0; i < tesla_dev_.n_streams; i++) {
      CUDA_SAFE_CALL( cudaStreamDestroy(tesla_dev_.streams[i]) );
    }
  }
}; // end of class Vectormap_CUDA_Base 

template<int _DIM, typename _FP, typename _BT, typename _BT_ATTR, typename _CELL_ATTR>
struct Vectormap_CUDA_Simple : public Vectormap_CUDA_Base<_DIM, _FP, _BT, _BT_ATTR, _CELL_ATTR> {
  using Body = _BT;
  using BodyAttr = _BT_ATTR;
  using CellAttr = _CELL_ATTR;
  using Base = Vectormap_CUDA_Base<_DIM, _FP, _BT, _BT_ATTR, _CELL_ATTR>;
  
  /* (One argument mapping) */

  /* NOTE IT RUNS ON CPUs.  The kernel "tapas_kernel::L2P()" is not
     coded to be run on GPUs, since it accesses the cell. */

#if 1
  template <class Funct, class Cell, class... Args>
  void map1(Funct f, BodyIterator<Cell> iter, Args... args) {
    //std::cout << "Vectormap_CUDA_Simple::map1() is called. " << iter.size() << std::endl;
    int sz = iter.size();
    for (int i = 0; i < sz; i++) {
      f(*(iter + i), args...);
    }
  }

#else

  template <class Funct, class Cell, class... Args>
  void map1(Funct f, BodyIterator<Cell> b0, Args... args) {
    static std::mutex mutex0;
    static struct cudaFuncAttributes tesla_attr0;

    TESLA &dev = Base::tesla_dev();
    
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
    int s = (streamid % dev.n_streams);
    vectormap_cuda_kernel1<<<nblocks, ctasize, 0, dev.streams[s]>>>
      (b0, n0, f, args...);
  }
#endif

  /**
   * \brief Two argument mapping
   * Implements a map on a GPU.  It extracts vectors of bodies.  It
   * uses a fixed command stream to serialize processing on each cell.
   * A call to cudaDeviceSynchronize() is needed on the caller of
   * Tapas-map.  The CTA size is the count in the first cell rounded
   * up to multiples of 256.  The tile size is the count in the first
   * cell rounded down to multiples of 64 (tile size is the count of
   * preloading of the second cells). 
   */
  template <class Funct, class Cell, class... Args>
  void Plain2(Funct f, Cell &c0, Cell &c1, Args... args) {
    static_assert(std::is_same<Body, typename Cell::BT::type>::value, "inconsistent template arguments");
    static_assert(std::is_same<BodyAttr, typename Cell::BT_ATTR>::value, "inconsistent template arguments");
                   
    using BT = Body;
    using BT_ATTR = BodyAttr;

    // nvcc's bug? the compiler cannot find base class' member function
    // so we need "Base::"
    TESLA &dev = Base::tesla_dev();

    static std::mutex mutex1;
    static struct cudaFuncAttributes tesla_attr1;
    if (tesla_attr1.binaryVersion == 0) {
      mutex1.lock();
      CUDA_SAFE_CALL(cudaFuncGetAttributes(
          &tesla_attr1,
          &vectormap_cuda_plain_kernel2<Funct, BT, BT_ATTR, Args...>));
      mutex1.unlock();
    }
    assert(tesla_attr1.binaryVersion != 0);

    assert(c0.IsLeaf() && c1.IsLeaf());
    /* (Cast to drop const, below). */
    BT* v0 = (BT*)&(c0.body(0));
    BT* v1 = (BT*)&(c1.body(0));
    BT_ATTR* a0 = (BT_ATTR*)&(c0.body_attr(0));
    size_t n0 = c0.nb();
    size_t n1 = c1.nb();
    assert(n0 != 0 && n1 != 0);

    /*bool am = AllowMutual<T1_Iter, T2_Iter>::value(b0, b1);*/
    /*int n0up = (TAPAS_CEILING(n0, 256) * 256);*/
    /*int n0up = (TAPAS_CEILING(n0, 32) * 32);*/
    int cta0 = (TAPAS_CEILING(dev.cta_size, 32) * 32);
    int ctasize = std::min(cta0, tesla_attr1.maxThreadsPerBlock);
    assert(ctasize == dev.cta_size);

    int tile0 = (dev.scratchpad_size / sizeof(Body));
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

    int s = (((unsigned long)&c0 >> 4) % dev.n_streams);
    vectormap_cuda_plain_kernel2<<<nblocks, ctasize, scratchpadsize,
        dev.streams[s]>>>
        (v0, v1, a0, n0, n1, tilesize, f, args...);
  } // end of void Plain2
  
  /** 
   * \fn Vectormap_CUDA_Simple::map2
   * \brief Calls a function FN given by the user on each data pair in the
   *        cells.  f takes arguments of Body&, Body&,
   *        BodyAttr&, and extra call arguments. 
   */
  template <class Funct, class Cell, class...Args>
  void map2(Funct f, ProductIterator<BodyIterator<Cell>> prod,
                   Args... args) {
    //printf("Vectormap_CUDA_Simple::map2\n"); fflush(0);
    
    typedef BodyIterator<Cell> Iter;
    const Cell &c0 = prod.first().cell();
    const Cell &c1 = prod.second().cell();
    if (c0 == c1) {
      Plain2(f, c0, c1, args...);
    } else {
      Plain2(f, c0, c1, args...);
      //Plain2(f, c1, c0, args...); // mutual is not supported 
    }
  }
}; // end of class Vectormap_CUD_Simple

template <class T>
struct Cell_Data {
  int size;
  T* data;
};

/**
 * \brief CUDA kernel invoke for 2-parameter Map()
 * Launches a kernel on Tesla.
 * Used by Vectormap_CUDA_Pakced and Applier.
 */
template <class Caller, class Funct, class... Args>
void invoke2(Caller *caller, int start, int nc, Cell_Data<Body> &r,
             int tilesize, size_t nblocks, int ctasize, int scratchpadsize,
             Funct f, Args... args) {
  using BV = typename Caller::Body;
  using BA = typename Caller::BodyAttr;

  TESLA &tesla_dev = caller->tesla_dev();

  /*AHO*/
  if (0) {
    printf("kernel(nblocks=%ld ctasize=%d scratchpadsize=%d tilesize=%d)\n",
           nblocks, ctasize, scratchpadsize, tilesize);
    printf("invoke(start=%d ncells=%d)\n", start, nc);

    for (int i = 0; 0 && i < nc; i++) {
      Cell_Data<BV> &lc = std::get<0>(caller->cellpairs2_[start + i]);
      Cell_Data<BA> &ac = std::get<1>(caller->cellpairs2_[start + i]);
      Cell_Data<BV> &rc = std::get<2>(caller->cellpairs2_[start + i]);
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
      (&(caller->body_list2_.ddata[start]), &(caller->attr_list2_.ddata[start]),
       nc, r.size, r.data,
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

/**
 * \brief Abstract base class of Applier.
 * 
 * Subclasses take a particular function type (ex. P2P) and variadic arguments.
 * When map-1 and map-2 are to be executed on GPUs, there are two steps:
 *   1. When user's function applies Map() over bodies or product of (bodies x bodies), 
 *      the pairs and the callback functions are joined to a interaction list.
 *   2. After all the traveresals are done, GPU invokes the callback function(s) over the
 *      interaction list.
 * The problem here is that Tapas has to save the function, not only the cell pairs.
 * In addition, functions should NOT be wrapped as std::function because it prevents the compiler
 * from inlining. Functions must be held statically. 
 * A subclass of AbstractApplier is a template class with a parameter of `class Funct` and 
 * they hold the callback function and variadic parameters.
 */
template<class Vectormap>
class AbstractApplier {
 public:
  virtual void apply(Vectormap *vm) = 0;
  virtual ~AbstractApplier() { }
};

/**
 * Applier for Map-2 (2-parameter map)
 */
// When tapas::Map <Body x Body> (corresponds to Map-2)
template<class Vectormap, class Funct, class...Args>
class Applier2 : public AbstractApplier<Vectormap> {
  Funct f_;
  std::tuple<Args...> args_; // variadic arguments
  std::mutex mutex_;
  cudaFuncAttributes func_attrs_;
  
  using ParamIdxSeq = typename gens<sizeof...(Args)>::type; // used to hold args... for invoke

 public:

  using Body = typename Vectormap::Body;
  using BodyAttr = typename Vectormap::BodyAttr;

  // Call ::invoke() function with args... 
  template<int ...ParamIdx>
  inline void invoke(Vectormap *caller, int start, int nc, Cell_Data<Body> &r,
                     int tilesize, size_t nblocks, int ctasize, int scratchpadsize,
                     seq<ParamIdx...>) {
    ::tapas::invoke2(caller, start, nc, r, tilesize, nblocks, ctasize, scratchpadsize, f_, std::get<ParamIdx>(args_)...);
  }

  // ctor. not thread safe.
  Applier2(Funct f, Args... args) : f_(f), args_(args...), func_attrs_() {
    using BV = Body;
    using BA = BodyAttr;
    if (func_attrs_.binaryVersion == 0) {
      CUDA_SAFE_CALL(cudaFuncGetAttributes(
          &func_attrs_,
          &vectormap_cuda_pack_kernel2<Funct, Body, BodyAttr, Cell_Data, Args...>));

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

    using MapData2 = typename Vectormap::MapData2;

    TESLA &tesla_dev = vm->tesla_dev();
    
    assert(vm->cellpairs2_.size() != 0);
#ifdef TAPAS_DEBUG
    printf(";; pairs=%ld\n", vm->cellpairs2_.size());
#endif
    

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

    size_t nn = vm->cellpairs2_.size();
    vm->body_list2().assure_size(nn);
    vm->attr_list2().assure_size(nn);

    auto t1 = high_resolution_clock::now();

    auto comp = cellcompare_r<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>(); // compare func
    std::sort(vm->cellpairs2_.begin(), vm->cellpairs2_.end(), comp);
    
    for (size_t i = 0; i < nn; i++) {
      MapData2 &c = vm->cellpairs2_[i];
      vm->body_list2().hdata[i] = std::get<0>(c);
      vm->attr_list2().hdata[i] = std::get<1>(c);
    }
    
    vm->body_list2().copy_in(nn);
    vm->attr_list2().copy_in(nn);

    auto t2 = high_resolution_clock::now();
    
    Cell_Data<BV> xr = std::get<2>(vm->cellpairs2_[0]);
    int xncells = 0;
    int xndata = 0;
    
    for (size_t i = 0; i < nn; i++) {
      MapData2 &c = vm->cellpairs2_[i];
      Cell_Data<Body> &r = std::get<2>(c);
      if (xr.data != r.data) {
        assert(i != 0 && xncells > 0);
        this->invoke(vm, (i - xncells), xncells, xr,
                     tilesize, nblocks, ctasize, scratchpadsize,
                     ParamIdxSeq());
        xncells = 0;
        xndata = 0;
        xr = r;
      }
      Cell_Data<Body> &l = std::get<0>(c);
      size_t nb = TAPAS_CEILING((xndata + l.size), ctasize);
      //std::cerr << "nb = " << nb << ", nblocks = " << nblocks << std::endl;
      if (nb > nblocks) {
        //std::cerr << "i = " << i << ", xncells = " << xncells << std::endl;
        assert(i != 0 && xncells > 0);
        this->invoke(vm, (i - xncells), xncells, xr,
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
    this->invoke(vm, (nn - xncells), xncells, xr,
                 tilesize, nblocks, ctasize, scratchpadsize,
                 ParamIdxSeq());
    
    // Report time (In a ad-hoc way using std::cout. Needs refactoring)
    double time_mcopy = duration_cast<microseconds>(t2-t1).count() * 1e-6;
  }

  

  virtual ~Applier2() { }
}; // class Applier2

template <typename T>
struct Mirror_Data {
  T* ddata; // device data
  T* hdata; // host data
  size_t size;

  Mirror_Data() : ddata(nullptr), hdata(nullptr), size(0) { }

  void assure_size(size_t n) {
    if (size < n) {
      free_data();
      size = n;
      cudaError_t ce;
      ce = cudaMalloc(&this->ddata, (sizeof(T) * n));
      assert(ce == cudaSuccess);
      ce = cudaMallocHost(&this->hdata, (sizeof(T) * n));
      assert(ce == cudaSuccess);
    }
  }

  void free_data() {
    cudaError_t ce;
    ce = cudaFree(this->ddata);
    assert(ce == cudaSuccess);
    ce = cudaFree(this->hdata);
    assert(ce == cudaSuccess);
  }

  void copy_in(size_t n) {
    assert(size == n);
    cudaError_t ce;
    ce = cudaMemcpy(this->ddata, this->hdata, (sizeof(T) * size),
                    cudaMemcpyHostToDevice);
    assert(ce == cudaSuccess);
  }
};

template<int _DIM, typename _FP, typename _BT, typename _BT_ATTR, typename _CELL_ATTR>
struct Vectormap_CUDA_Packed
    : public Vectormap_CUDA_Base<_DIM, _FP, _BT, _BT_ATTR, _CELL_ATTR> {
  using VectorMap = Vectormap_CUDA_Packed<_DIM, _FP, _BT, _BT_ATTR, _CELL_ATTR>; // self
  using BT = _BT;
  using BT_ATTR = _BT_ATTR;

  using Body = _BT;
  using BodyAttr = _BT_ATTR;

  using Body_List = Cell_Data<BT>;
  using Attr_List = Cell_Data<BT_ATTR>;
  using MapData2 = std::tuple<Body_List, Attr_List, Body_List>;

  // Data for 1-parameter Map()
  std::mutex pack1_mutex_; // mutex for map1
  
  // Data for 2-parameter Map()
  std::vector<MapData2> cellpairs2_;
  Mirror_Data<Body_List> body_list2_; // body list for 2-parameter Map()
  Mirror_Data<Attr_List> attr_list2_; // body attr list for 2-parameter Map()
  std::mutex pack2_mutex_; // mutex for map2
  
  // funct_id_ is used to check if the same Funct is used for all cell pairs.
  // In the current implementation, it is assumed that a single function and
  // the same optional arguments (= Args...) are used to all cell pairs.
  std::mutex applier2_mutex_;
  intptr_t funct_id_;
  AbstractApplier<VectorMap> *applier2_;
  
  cudaFuncAttributes func_attrs_;

  double time_device_call_;
  
  void Start2() {
#ifdef TAPAS_DEBUG
    printf(";; start\n"); fflush(0);
#endif
    cellpairs2_.clear();
    //cellpairs1_.clear();
  }

  /**
   * @brief ctor.
   * not thread safe
   */
  Vectormap_CUDA_Packed()
      : cellpairs2_()
      , body_list2_()
      , attr_list2_()
      , pack1_mutex_()
      , pack2_mutex_()
      , applier2_mutex_()
      , funct_id_(0)
      , applier2_(nullptr)
      , time_device_call_(0)
  { }

  inline std::vector<MapData2> &cellpairs2() {
    return cellpairs2_;
  }

  inline Mirror_Data<Body_List> &body_list2() {
    return body_list2_;
  }
  
  inline Mirror_Data<Attr_List> &attr_list2() {
    return attr_list2_;
  }

  template <class Funct, class Cell, class... Args>
  inline void map1(Funct f, BodyIterator<Cell> iter, Args... args) {
    //std::cout << "Yey! new Vectormap_CUDA_Packed::Map1() is called. " << iter.size() << std::endl;
    int sz = iter.size();
    for (int i = 0; i < sz; i++) {
      f(*iter, iter.attr(), args...);
      iter++;
    }
  }

  /* (Two argument mapping with left packing.) */

  /**
   * \brief Vectormap_CUDA_Packed::map2
   */
  template <class Cell, class Funct, class... Args>
  void map2(Funct f, ProductIterator<BodyIterator<Cell>> prod, Args... args) {
    static_assert(std::is_same<typename Cell::Body, Body>::value, "inconsistent Cell and Body types");
    static_assert(std::is_same<typename Cell::BodyAttr, BodyAttr>::value, "inconsistent Cell and BodyAttr types");

    const Cell &c0 = prod.first().cell();
    const Cell &c1 = prod.second().cell();
    assert(c0.IsLeaf() && c1.IsLeaf());

    if (c0.nb() == 0 || c1.nb() == 0) return;

    // Create Applier with Funct and Args...
    if (applier2_ == nullptr) {
      applier2_mutex_.lock();
      if (applier2_ == nullptr) {
        applier2_ = new Applier2<VectorMap, Funct, Args...>(f, args...);
        funct_id_ = Type2Int<Funct>::value();

        // Memo [Jan 18, 2016]
        // func_id_ is not used as of now. This check value is for when there are multiple kernels
        // for bodies x bodies product map.
        // An interaction list is created for each function (stored in a unordered_map of which keys are integers).
      }
      applier2_mutex_.unlock();
    }
    
    TAPAS_ASSERT(funct_id_ == Type2Int<Funct>::value());

    /* (Cast to drop const, below). */
    Cell_Data<BT> d0;
    Cell_Data<BT> d1;
    Cell_Data<BT_ATTR> a0;
    //Cell_Data<BT_ATTR> a1;
    d0.size = c0.nb();
    d0.data = (BT*)&(c0.body(0));
    a0.size = c0.nb();
    a0.data = (BT_ATTR*)&(c0.body_attr(0));
    d1.size = c1.nb();
    d1.data = (BT*)&(c1.body(0));
    //a1.size = c1.nb();
    //a1.data = (BT_ATTR*)&(c1.body_attr(0)); // unused?
    
    if (c0 == c1) {
      pack2_mutex_.lock();
      cellpairs2_.push_back(MapData2(d0, a0, d1));
      pack2_mutex_.unlock();
    } else {
      pack2_mutex_.lock();
      cellpairs2_.push_back(MapData2(d0, a0, d1));
      // mutual interaction is not supported in this CUDA version.
      //cellpairs2_.push_back(std::tuple<Cell_Data<BT>, Cell_Data<BT_ATTR>, Cell_Data<BT>>(d1, a1, d0)); // mutual is not supported in CUDA version
      pack2_mutex_.unlock();
    }
  }
  
  /* Limit of the number of threads in grids. */

  static const constexpr int N0 = (16 * 1024);

  /* Starts launching a kernel on collected cells. */
  
  void on_collected2() {
    auto t1 = std::chrono::high_resolution_clock::now();

    TAPAS_ASSERT(applier2_ != nullptr);
    
    applier2_->apply(this);

    vectormap_check_error("Vectormap_CUDA_Packed::end", __FILE__, __LINE__);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();
    
    time_device_call_ = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() * 1e-6;
  }
  
  void Finish2() {
#ifdef TAPAS_DEBUG
    printf(";; Vectormap_CUDA_Packed::Finish2\n"); fflush(0);
#endif
    on_collected2();
    
    if (applier2_ != nullptr) {
      applier2_mutex_.lock();
      
      // record time
      
      if (applier2_ != nullptr) {
        delete applier2_;
        applier2_ = nullptr;
        funct_id_ = 0;
      }
      applier2_mutex_.unlock();
    }
  }
}; // Vectormap_CUDA_Packed

} //namespace tapas

#endif /*__CUDACC__*/

#endif /*TAPAS_VECTORMAP_CUDA_H_*/
