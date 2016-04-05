#ifndef __TAPAS_HOT_LET_COMMON__
#define __TAPAS_HOT_LET_COMMON__

namespace tapas {
namespace hot {

/**
 * Enum values of predicate function
 */
enum class SplitType {
  Approx,       // Compute using right cell's attribute
  Body,         // Compute using right cell's bodies
  SplitLeft,    // Split Left (local) cell
  SplitRight,   // Split Right (remote) cell
  SplitBoth,    // Split both cells
  None,         // Nothing. Use when a target cell isn't local in Traverse
};

} /* namespace hot */
} /* namespace tapas */

#endif /* __TAPAS_HOT_LET_COMMON__ */
