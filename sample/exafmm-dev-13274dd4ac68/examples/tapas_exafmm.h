#ifndef EXAFMM_TAPAS_H_
#define EXAFMM_TAPAS_H_

#include "types.h" // exafmm/include/types.h
#include "tapas/debug_util.h"
#include "tapas/util.h"

#include "tapas.h"

struct CellAttr {
    real_t R;
    vecP M;
    vecP L;
};

// Body is defined in types.h
using BodyAttr = kvec4;

template<class T, class U>
constexpr size_t MemberOffset(U T::*pmem) {
  return (char*)&((T*)nullptr->*pmem) - (char*)nullptr;
}

//
// Offset of coordinate values in Body
// ex.
// If body type is
//   struct MyBody {
//     vec3 X;
//   };
// then it's 0.
//
#ifdef TAPAS_COMPILER_CLANG
// The MemberOffset functions is not constepxr in clang.
const constexpr size_t kBodyCoordOffset = 0;
#else
const constexpr size_t kBodyCoordOffset = MemberOffset(&Body::X);
#endif

// Select MPI or single-process
#ifdef USE_MPI
#include "tapas/hot.h"
#else
#include "tapas/single_node_hot.h"
#endif /* USE_MPI */

#ifdef MTHREADS
#warning "MTHREADS is defined. Do you mean \"MTHREAD\"?"
#endif

// Select threading component: serial/MassiveThreads/TBB
#if defined(MTHREAD)

#include "tapas/threading/massivethreads.h"
using FMM_Threading = tapas::threading::MassiveThreads;

#elif defined(TBB)

#include "tapas/threading/tbb.h"
using FMM_Threading = tapas::threading::IntelTBB;

#else

#include "tapas/threading/serial.h"
using FMM_Threading = tapas::threading::Serial;

#endif

struct FMM_Params : public tapas::HOT<3, real_t, Body, kBodyCoordOffset, kvec4, CellAttr> {
  using Threading = FMM_Threading;
};

using TapasFMM = tapas::Tapas<FMM_Params>;

typedef TapasFMM::Region Region;

#endif // EXAFMM_TAPAS_H_
