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

static struct TESLA {
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
} tesla_dev;

#define TAPAS_CEILING(X,Y) (((X) + (Y) - 1) / (Y))
#define TAPAS_FLOOR(X,Y) ((X) / (Y))

static void vectormap_check_error(const char *msg, const char *file, const int line) {
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
static double atomicAdd(float* address, float val) {
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
  BA q0 = {0.0f, 0.0f, 0.0f, 0.0f};

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
        f(&p0, &p1, q0, args...);
      }
    }
    __syncthreads();
  }

  if (cell != -1) {
    assert(item < a[cell].size);
    BA &a0 = a[cell].data[item];
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

  /** Memory allocator for the unified memory.  It will replace the
      vector allocators.  (N.B. Its name should be generic because it
      is used in CPUs also.) */

  template <typename T>
  struct um_allocator : public std::allocator<T> {
  public:
    /*typedef T* pointer;*/
    /*typedef const T* const_pointer;*/
    /*typedef T value_type;*/
    template <class U> struct rebind {typedef um_allocator<U> other;};

    T* allocate(size_t n, const void* hint = 0) {
      T* p;
      cudaError_t ce;
      ce = cudaMallocManaged(&p, (sizeof(T) * n), cudaMemAttachGlobal);
      assert(ce == cudaSuccess && p != 0);
      fprintf(stderr, ";; cudaMallocManaged() p=%p n=%zd\n", p, n); fflush(0);
      return p;
    }

    void deallocate(T* p, size_t n) {
      cudaError_t ce = cudaFree(p);
      assert(ce == cudaSuccess);
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

  static void vectormap_setup(int cta, int nstreams) {
    assert(nstreams <= TAPAS_CUDA_MAX_NSTREAMS);

    tesla_dev.cta_size = cta;
    tesla_dev.n_streams = nstreams;

    /*AHO*/ /* USE PROPER WAY TO KNOW OF USE OF MPI. */

#ifdef EXAFMM_TAPAS_MPI

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rankofnode, rankinnode, nprocsinnode;
    rank_in_node(MPI_COMM_WORLD, rankofnode, rankinnode, nprocsinnode);

#else /*EXAFMM_TAPAS_MPI*/

    int rank = 0;
    int rankinnode = 0;
    int nprocsinnode = 1;

#endif /*EXAFMM_TAPAS_MPI*/

    cudaError_t ce;
    int ngpus;
    ce = cudaGetDeviceCount(&ngpus);
    assert(ce == cudaSuccess);
    if (ngpus < nprocsinnode) {
      fprintf(stderr, "More ranks than GPUs on a node\n");
      assert(ngpus >= nprocsinnode);
    }

    tesla_dev.gpuno = rankinnode;
    cudaDeviceProp prop;
    ce = cudaGetDeviceProperties(&prop, tesla_dev.gpuno);
    assert(ce == cudaSuccess);
    ce = cudaSetDevice(tesla_dev.gpuno);
    assert(ce == cudaSuccess);

    printf(";; Rank#%d uses GPU#%d\n", rank, tesla_dev.gpuno);

    assert(prop.unifiedAddressing);

    tesla_dev.sm = (prop.major * 10 + prop.minor);
    tesla_dev.n_sm = prop.multiProcessorCount;

    tesla_dev.n_cores = 0;
    for (struct TESLA_CORES &i : tesla_cores) {
      if (i.sm == tesla_dev.sm) {
        tesla_dev.n_cores = i.n_cores;
        break;
      }
    }
    assert(tesla_dev.n_cores != 0);

    tesla_dev.scratchpad_size = prop.sharedMemPerBlock;
    tesla_dev.max_cta_size = prop.maxThreadsPerBlock;
    assert(prop.maxThreadsPerMultiProcessor >= prop.maxThreadsPerBlock * 2);

    for (int i = 0; i < tesla_dev.n_streams; i++) {
      ce = cudaStreamCreate(&tesla_dev.streams[i]);
      assert(ce == cudaSuccess);
    }
  }

  static void vectormap_release() {
    for (int i = 0; i < tesla_dev.n_streams; i++) {
      cudaError_t ce = cudaStreamDestroy(tesla_dev.streams[i]);
      assert(ce == cudaSuccess);
    }
  }

  static void vectormap_start() {}

  template <class Funct, class... Args>
  static void vectormap_finish(Funct f, Args... args) {
    vectormap_check_error("vectormap_end", __FILE__, __LINE__);
    cudaError_t ce;
    ce = cudaDeviceSynchronize();
    if (ce != cudaSuccess) {
      fprintf(stderr,
              "%s:%i (%s): CUDA ERROR (%d): %s.\n",
              __FILE__, __LINE__, "cudaDeviceSynchronize", (int)ce, cudaGetErrorString(ce));
      assert(ce == cudaSuccess);
    }
  }

  /* (One argument mapping) */

  /* NOTE IT RUNS ON CPUs.  The kernel "tapas_kernel::L2P()" is not
     coded to be run on GPUs, since it accesses the cell. */

  template <class Funct, class Cell, class... Args>
  static void vector_map1(Funct f, BodyIterator<Cell> iter,
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
      cudaError_t ce = cudaFuncGetAttributes(
        &tesla_attr0,
        &vectormap_cuda_kernel1<Funct, typename Cell::BodyType, Args...>);
      assert(ce == cudaSuccess);
      mutex0.unlock();
    }
    assert(tesla_attr0.binaryVersion != 0);

    size_t n0 = b0.size();
    int n0up = (TAPAS_CEILING(n0, 256) * 256);
    int ctasize = std::min(n0up, tesla_attr0.maxThreadsPerBlock);
    size_t nblocks = TAPAS_CEILING(n0, ctasize);

    streamid++;
    int s = (streamid % tesla_dev.n_streams);
    vectormap_cuda_kernel1<<<nblocks, ctasize, 0, tesla_dev.streams[s]>>>
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
  static void vectormap_cuda_plain(Funct f, Cell &c0, Cell &c1,
                                   Args... args) {
    typedef typename Cell::BT::type BV;
    typedef typename Cell::BT_ATTR BA;

    static std::mutex mutex1;
    static struct cudaFuncAttributes tesla_attr1;
    if (tesla_attr1.binaryVersion == 0) {
      mutex1.lock();
      cudaError_t ce = cudaFuncGetAttributes(
        &tesla_attr1,
        &vectormap_cuda_plain_kernel2<Funct, BV, BA, Args...>);
      assert(ce == cudaSuccess);
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
    int cta0 = (TAPAS_CEILING(tesla_dev.cta_size, 32) * 32);
    int ctasize = std::min(cta0, tesla_attr1.maxThreadsPerBlock);
    assert(ctasize == tesla_dev.cta_size);

    int tile0 = (tesla_dev.scratchpad_size / sizeof(typename Cell::BodyType));
    int tile1 = (TAPAS_FLOOR(tile0, 32) * 32);
    int tilesize = std::min(ctasize, tile1);
    assert(tilesize > 0);

    int scratchpadsize = (sizeof(typename Cell::BodyType) * tilesize);
    size_t nblocks = TAPAS_CEILING(n0, ctasize);

#if 0 /*AHO*/
    fprintf(stderr, "launch arrays=(%p/%ld, %p/%ld, %p/%ld) blks=%ld cta=%d\n",
            v0, n0, v1, n1, a0, n0, nblocks, ctasize);
    fflush(0);
#endif

    int s = (((unsigned long)&c0 >> 4) % tesla_dev.n_streams);
    vectormap_cuda_plain_kernel2<<<nblocks, ctasize, scratchpadsize,
      tesla_dev.streams[s]>>>
      (v0, v1, a0, n0, n1, tilesize, f, args...);
  }

  /** Calls a function FN given by the user on each data pair in the
      cells.  FN takes arguments of Cell::BodyType&, Cell::BodyType&,
      Cell::BodyAttrType&, and extra call arguments. */

  template <class Funct, class Cell, class...Args>
  static void vector_map2(Funct f, ProductIterator<BodyIterator<Cell>> prod,
                          Args... args) {
    printf("vector_map2X\n"); fflush(0);

    typedef BodyIterator<Cell> Iter;
    const Cell &c0 = prod.first().cell();
    const Cell &c1 = prod.second().cell();
    if (c0 == c1) {
      vectormap_cuda_plain(f, c0, c1, args...);
    } else {
      vectormap_cuda_plain(f, c0, c1, args...);
      vectormap_cuda_plain(f, c1, c0, args...);
    }
  }

};

template <class T>
struct Cell_Data {
  int size;
  T* data;
};

template<int _DIM, class _FP, class _BT, class _BT_ATTR>
struct Vectormap_CUDA_Packed
  : Vectormap_CUDA_Simple<_DIM, _FP, _BT, _BT_ATTR> {
  typedef typename _BT::type BV;
  typedef _BT_ATTR BA;

  /* STATIC MEMBER FIELDS. (It is a trick.  See:
     http://stackoverflow.com/questions/11709859/) */

  static std::vector<std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>>
    &cellpairs() {
    static std::vector<std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>>
      cellpairs_;
    return cellpairs_;
  }

  static std::mutex &pairs_mutex() {
    static std::mutex pairs_mutex_;
    return pairs_mutex_;
  }

  static size_t &npairs() {
    static size_t npairs_ = 0;
    return npairs_;
  }

  static Cell_Data<BV>* &dvcells() {
    static Cell_Data<BV>* dvcells_ = 0;
    return dvcells_;
  }

  static Cell_Data<BV>* &hvcells() {
    static Cell_Data<BV>* hvcells_ = 0;
    return hvcells_;
  }

  static Cell_Data<BA>* &dacells() {
    static Cell_Data<BA>* dacells_ = 0;
    return dacells_;
  }

  static Cell_Data<BA>* &hacells() {
    static Cell_Data<BA>* hacells_ = 0;
    return hacells_;
  }

  static void vectormap_start() {
    //printf(";; vectormap_start\n"); fflush(0);
    cellpairs().clear();
  }

  /* (Two argument mapping with left packing.) */

  template <class Funct, class Cell, class... Args>
  static void vector_map2(Funct f, ProductIterator<BodyIterator<Cell>> prod,
                          Args... args) {
    typedef typename Cell::BT::type BV;
    typedef typename Cell::BT_ATTR BA;

    const Cell &c0 = prod.first().cell();
    const Cell &c1 = prod.second().cell();
    assert(c0.IsLeaf() && c1.IsLeaf());

    if (c0.nb() == 0 || c1.nb() == 0) return;
    
    /* (Cast to drop const, below). */
    Cell_Data<BV> d0;
    Cell_Data<BV> d1;
    Cell_Data<BA> a0;
    Cell_Data<BA> a1;
    d0.size = c0.nb();
    d0.data = (BV*)&(c0.body(0));
    a0.size = c0.nb();
    a0.data = (BA*)&(c0.body_attr(0));
    d1.size = c1.nb();
    d1.data = (BV*)&(c1.body(0));
    a1.size = c1.nb();
    a1.data = (BA*)&(c1.body_attr(0));
    if (c0 == c1) {

      pairs_mutex().lock();
      cellpairs().push_back(std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>(d0, a0, d1));
      pairs_mutex().unlock();
    } else {
      pairs_mutex().lock();
      cellpairs().push_back(std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>(d0, a0, d1));
      cellpairs().push_back(std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>(d1, a1, d0));
      pairs_mutex().unlock();
    }
  }

  /* Launches a kernel on Tesla. */

  template <class Funct, class Cell, class... Args>
  static void vectormap_invoke(int start, int nc, Cell_Data<BV> &r,
                               int tilesize,
                               size_t nblocks, int ctasize, int scratchpadsize,
                               Cell &dummy, Funct f, Args... args) {
    typedef typename Cell::BT::type BV;
    typedef typename Cell::BT_ATTR BA;

    /*AHO*/
    if (0) {
      printf("kernel(nblocks=%ld ctasize=%d scratchpadsize=%d tilesize=%d\n",
             nblocks, ctasize, scratchpadsize, tilesize);
      printf("invoke(start=%d ncells=%d)\n", start, nc);
      for (int i = 0; i < nc; i++) {
        Cell_Data<BV> &lc = std::get<0>(cellpairs()[start + i]);
        Cell_Data<BA> &ac = std::get<1>(cellpairs()[start + i]);
        Cell_Data<BV> &rc = std::get<2>(cellpairs()[start + i]);
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
      (&(dvcells()[start]), &(dacells()[start]), nc, r.size, r.data,
       tilesize, f, args...);
  }

  /* Limit of the number of threads in grids. */

  static const int N0 = (16 * 1024);

  /* Starts launching a kernel on collected cells. */

  template <class Funct, class Cell, class... Args>
  static void vectormap_on_collected(Funct f, Cell dummy, Args... args) {
    typedef typename Cell::BT::type BV;
    typedef typename Cell::BT_ATTR BA;

    if (cellpairs().size() == 0) {
      return;
    }
    cudaError_t ce;
    static std::mutex mutex2;
    static struct cudaFuncAttributes tesla_attr2;
    if (tesla_attr2.binaryVersion == 0) {
      mutex2.lock();
      cudaError_t ce = cudaFuncGetAttributes(
        &tesla_attr2,
        &vectormap_cuda_pack_kernel2<Funct, BV, BA, Cell_Data, Args...>);
      assert(ce == cudaSuccess);
      mutex2.unlock();
      if (0) {
        /*GOMI*/
        printf((";; vectormap_cuda_pack_kernel2:"
                " binaryVersion=%d, cacheModeCA=%d, constSizeBytes=%zd,"
                " localSizeBytes=%zd, maxThreadsPerBlock=%d, numRegs=%d,"
                " ptxVersion=%d, sharedSizeBytes=%zd\n"),
               tesla_attr2.binaryVersion, tesla_attr2.cacheModeCA,
               tesla_attr2.constSizeBytes, tesla_attr2.localSizeBytes,
               tesla_attr2.maxThreadsPerBlock, tesla_attr2.numRegs,
               tesla_attr2.ptxVersion, tesla_attr2.sharedSizeBytes);
        fflush(0);
      }
    }
    assert(tesla_attr2.binaryVersion != 0);

    //printf(";; pairs=%ld\n", cellpairs().size());

    int cta0 = (TAPAS_CEILING(tesla_dev.cta_size, 32) * 32);
    int ctasize = std::min(cta0, tesla_attr2.maxThreadsPerBlock);
    assert(ctasize == tesla_dev.cta_size);

    int tile0 = (tesla_dev.scratchpad_size / sizeof(typename Cell::BodyType));
    int tile1 = (TAPAS_FLOOR(tile0, 32) * 32);
    int tilesize = std::min(ctasize, tile1);
    assert(tilesize > 0);

    int scratchpadsize = (sizeof(typename Cell::BodyType) * tilesize);
    size_t nblocks = TAPAS_CEILING(N0, ctasize);

    if (npairs() < cellpairs().size()) {
      ce = cudaFree(dvcells());
      assert(ce == cudaSuccess);
      ce = cudaFree(dacells());
      assert(ce == cudaSuccess);
      ce = cudaFree(hvcells());
      assert(ce == cudaSuccess);
      ce = cudaFree(hacells());
      assert(ce == cudaSuccess);

      npairs() = cellpairs().size();
      ce = cudaMalloc(&dvcells(), (sizeof(Cell_Data<BV>) * npairs()));
      assert(ce == cudaSuccess);
      ce = cudaMalloc(&dacells(), (sizeof(Cell_Data<BA>) * npairs()));
      assert(ce == cudaSuccess);
      ce = cudaMallocHost(&hvcells(), (sizeof(Cell_Data<BV>) * npairs()));
      assert(ce == cudaSuccess);
      ce = cudaMallocHost(&hacells(), (sizeof(Cell_Data<BA>) * npairs()));
      assert(ce == cudaSuccess);
    }

    std::sort(cellpairs().begin(), cellpairs().end(),
              cellcompare_r<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>>());
    for (size_t i = 0; i < npairs(); i++) {
      std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>> &c = cellpairs()[i];
      hvcells()[i] = std::get<0>(c);
    }
    ce = cudaMemcpy(dvcells(), hvcells(), (sizeof(Cell_Data<BV>) * npairs()),
                    cudaMemcpyHostToDevice);
    assert(ce == cudaSuccess);
    for (size_t i = 0; i < npairs(); i++) {
      std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>> &c = cellpairs()[i];
      hacells()[i] = std::get<1>(c);
    }
    ce = cudaMemcpy(dacells(), hacells(), (sizeof(Cell_Data<BA>) * npairs()),
                    cudaMemcpyHostToDevice);
    assert(ce == cudaSuccess);

    Cell_Data<BV> xr = std::get<2>(cellpairs()[0]);
    int xncells = 0;
    int xndata = 0;
    for (size_t i = 0; i < npairs(); i++) {
      std::tuple<Cell_Data<BV>, Cell_Data<BA>, Cell_Data<BV>> &c = cellpairs()[i];
      Cell_Data<BV> &r = std::get<2>(c);
      if (xr.data != r.data) {
        assert(i != 0 && xncells > 0);
        vectormap_invoke((i - xncells), xncells, xr,
                         tilesize, nblocks, ctasize, scratchpadsize,
                         dummy, f, args...);
        xncells = 0;
        xndata = 0;
        xr = r;
      }
      Cell_Data<BV> &l = std::get<0>(c);
      size_t nb = TAPAS_CEILING((xndata + l.size), ctasize);
      if (nb > nblocks) {
        assert(i != 0 && xncells > 0);
        vectormap_invoke((i - xncells), xncells, xr,
                         tilesize, nblocks, ctasize, scratchpadsize,
                         dummy, f, args...);
        xncells = 0;
        xndata = 0;
        xr = r;
      }
      xncells++;
      xndata += (TAPAS_CEILING(l.size, 32) * 32);
    }
    assert(xncells > 0);
    vectormap_invoke((npairs() - xncells), xncells, xr,
                     tilesize, nblocks, ctasize, scratchpadsize,
                     dummy, f, args...);
  }

  template <class Funct, class Cell, class... Args>
  static void vectormap_finish(Funct f, Cell dummy, Args... args) {
    //printf(";; vectormap_finish\n"); fflush(0);
    vectormap_on_collected(f, dummy, args...);
    vectormap_check_error("vectormap_end", __FILE__, __LINE__);
    cudaError_t ce;
    ce = cudaDeviceSynchronize();
    if (ce != cudaSuccess) {
      fprintf(stderr,
              "%s:%i (%s): CUDA ERROR (%d): %s.\n",
              __FILE__, __LINE__, "cudaDeviceSynchronize", (int)ce, cudaGetErrorString(ce));
      assert(ce == cudaSuccess);
    }
  }
};

}

#endif /*__CUDACC__*/

#endif /*TAPAS_VECTORMAP_CUDA_H_*/
