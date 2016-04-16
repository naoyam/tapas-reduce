#ifndef __TAPAS_HOT_LET_COMMON__
#define __TAPAS_HOT_LET_COMMON__

#include <string>

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

std::string ToString(SplitType st) {
  switch(st) {
    case SplitType::Approx:     return "SplitType::Approx";
    case SplitType::Body:       return "SplitType::Body";
    case SplitType::SplitLeft:  return "SplitType::SplitLeft";
    case SplitType::SplitRight: return "SplitType::SplitRight";
    case SplitType::SplitBoth:  return "SplitType::SplitBoth";
    case SplitType::None:       return "SplitType::None";
    default:
      assert(0);
      return "";
  }
}

std::ostream& operator<<(std::ostream &os, SplitType st) {
  os << ToString(st);
  return os;
}

} /* namespace hot */
} /* namespace tapas */

#endif /* __TAPAS_HOT_LET_COMMON__ */
