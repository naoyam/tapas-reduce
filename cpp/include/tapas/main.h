#ifndef TAPAS_MAIN_H_
#define TAPAS_MAIN_H_

#include "tapas/common.h"

namespace tapas {

/**
 * @brief Tapas' internal type used to wrap user-defined body type
 * @tparam BT Body type
 * @tparam POS_OFFSET Byte offset of coordinate values
 * Tapas assumes that n-dimensional coordinate values (x,y,z in 3-dim) are continuous.
 */
template <class BT, int POS_OFFSET>
class BodyInfo {
  public:
    typedef BT type;
    static const int pos_offset = POS_OFFSET;
};

} // namespace tapas


#endif // TAPAS_MAIN_H_ 
