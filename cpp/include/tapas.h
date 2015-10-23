#ifndef TAPAS_H_
#define TAPAS_H_

// Base header for user code. 

#include "tapas/vectormap.h"
#include "tapas/main.h"
#include "tapas/map.h"
#include "tapas/threading/default.h"
#include "tapas/vectormap_cpu.h"
#ifdef __CUDACC__
#include "tapas/vectormap_cuda.h"
#endif

#endif
