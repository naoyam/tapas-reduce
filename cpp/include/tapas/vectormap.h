/* vectormap.h -*- Coding: us-ascii-unix; -*- */
/* Copyright (C) 2015-2015 RIKEN AICS */

#ifndef TAPAS_VECTORMAP_H_
#define TAPAS_VECTORMAP_H_

/* Selector of Direct Map Implementations.  It selects either CPU or
   GPU by instantiating in TapasStaticParams.  It replaces lowest
   mapping tapas::Map(Funct f, ProductIterator<BodyIterator<Cell>>>
   prod, Args...args).  This file only includes forward references. */

#include <assert.h>
#include <memory>

namespace tapas {

#ifdef __CUDACC__

/** Memory allocator for the unified memory.  It will replace the
    vector allocators. */

template<int _DIM, typename _FP, typename _BT, typename _BT_ATTR, typename _CELL_ATTR>
struct Vectormap_CUDA_Simple;

template<int _DIM, typename _FP, typename _BT, typename _BT_ATTR, typename _CELL_ATTR>
struct Vectormap_CUDA_Packed;

#else /* else __CUDACC__ */

template<int _DIM, class _FP, class _BT, class _BT_ATTR, class _CELL_ATTR>
struct Vectormap_CPU;

#endif /*__CUDACC__*/

}

#endif /*TAPAS_VECTORMAP_H_*/
