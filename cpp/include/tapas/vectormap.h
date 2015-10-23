/* vectormap.h -*- Coding: us-ascii-unix; -*- */

#ifndef TAPAS_VECTORMAP_H_
#define TAPAS_VECTORMAP_H_

/* Selector of Direct Map Implementations.  It selects either CPU or
   GPU by instantiating in TapasStaticParams.  It replaces lowest
   mapping tapas::Map(Funct f, ProductIterator<BodyIterator<Cell>>>
   prod, Args...args).  This file only includes forward references. */

#include <assert.h>
#include <memory>

#define TAPAS_USE_VECTORMAP

namespace tapas {

#ifdef __CUDACC__

/** Memory allocator for the unified memory.  It will replace the
    vector allocators. */

template <typename T>
struct vectormap_allocator__ : public std::allocator<T> {
public:
  /*typedef T* pointer;*/
  /*typedef const T* const_pointer;*/
  /*typedef T value_type;*/
  template <class U> struct rebind {typedef vectormap_allocator__<U> other;};

  T* allocate(size_t n, const void* hint = 0) {
    T* p;
    cudaError_t ce;
    ce = cudaMallocManaged(&p, (sizeof(T) * n), cudaMemAttachGlobal);
    assert(ce == cudaSuccess && p != 0);
    fprintf(stderr, "cudaMallocManaged() p=%p n=%zd\n", p, n); fflush(0);
    return p;
  }

  void deallocate(T* p, size_t n) {
    cudaError_t ce = cudaFree(p);
    assert(ce == cudaSuccess);
    fprintf(stderr, "cudaFree() p=%p n=%zd\n", p, n); fflush(0);
  }

  explicit vectormap_allocator__() throw() : std::allocator<T>() {}

  explicit vectormap_allocator__(const vectormap_allocator__ &a) throw()
    : std::allocator<T>(a) {}

  template <class U> explicit
  vectormap_allocator__(const vectormap_allocator__<U> &a) throw()
    : std::allocator<T>(a) {}

  ~vectormap_allocator__() throw() {}
};

template<int _DIM, class _FP, class _BT, class _BT_ATTR>
struct Vectormap_CUDA_Simple;

template<int _DIM, class _FP, class _BT, class _BT_ATTR>
struct Vectormap_CUDA_Packed;

#else /*__CUDACC__*/

template <typename T>
using vectormap_allocator__ = std::allocator<T>;

template<int _DIM, class _FP, class _BT, class _BT_ATTR>
struct Vectormap_CPU;

#endif /*__CUDACC__*/

}

#endif /*TAPAS_VECTORMAP_H_*/
