#ifndef _TAPAS_THREADING_DEFAULT_H_
#define _TAPAS_THREADING_DEFAULT_H_

// Defined when non-default threading policy is included
// #define TAPAS_THREADING_NODEFAULT
// Here it is not defined since this file provides the default.
// If you provide non-default threading policy, define it.

#include "tapas/threading/serial.h"

namespace tapas {
namespace threading {

using Default = Serial;

} // namespace tapas
} // namespace thread

#endif // _TAPAS_THREADING_MASSIVETHREADS_H_

