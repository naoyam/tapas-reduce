#ifndef TAPAS_MORTON_COMMON_
#define TAPAS_MORTON_COMMON_

#include <cstdint>
#include <vector>
#include <algorithm>
#include <utility>

#include "tapas/vec.h"

/**
 * @file Common operations of Morton-key
 */

namespace tapas {
namespace morton_common {

/** 
 * @brief Morton key type.
 */
typedef uint64_t KeyType;

typedef std::list<KeyType> KeyList;
typedef std::vector<KeyType> KeyVector;
typedef std::unordered_set<KeyType> KeySet;
typedef std::pair<KeyType, KeyType> KeyPair;

using std::unordered_map;


// 
// Maximum depth of octree:
//   Maximum depth is determined by the type of KeyType and #bits of depth bits.
//
//   Type        | Depth bits | Max depth
//
//   uint32_t    |          3 |         4
//   uint32_t    |          4 |         8
//   uint32_t    |          5 |         8
//   uint32_t    |          6 |         8
//   uint32_t    |          7 |         8
//   uint32_t    |          8 |         7
//
//   uint64_t    |          3 |         4
//   uint64_t    |          4 |         8
//   uint64_t    |          5 |        16
//   uint64_t    |          6 |        19
//   uint64_t    |          7 |        18
//
//   __uint128_t |          4 |         8
//   __uint128_t |          5 |        16
//   __uint128_t |          6 |        32
//   __uint128_t |          7 |        40
//   __uint128_t |          8 |        39
//   __uint128_t |          9 |        39
//
// (__uint128_t is GCC's extension)

template<class __KeyType>
struct BestDepthBitWidth;

template<> struct BestDepthBitWidth<uint32_t> { static constexpr int Bits = 4; };
template<> struct BestDepthBitWidth<uint64_t> { static constexpr int Bits = 6; };
#if defined(__SIZEOF_INT128__) 
template<> struct BestDepthBitWidth<__uint128_t> { static constexpr int Bits = 7; };
#endif

/**
 * @brief Number of bits used to represent depth in a Morton key.
 */
const int DEPTH_BIT_WIDTH = BestDepthBitWidth<KeyType>::Bits;

#define MIN(a,b) ((a) > (b) ? (b) : (a))
const KeyType DEPTH_MASK = (1 << DEPTH_BIT_WIDTH) - 1;
const int MAX_DEPTH_BY_DEPTH_BITS = ((1 << DEPTH_BIT_WIDTH) - 1);
const int MAX_DEPTH_BY_KEY_BITS = (sizeof(KeyType) * 8 - 1 - DEPTH_BIT_WIDTH) / 3;
const int MAX_DEPTH = MIN ( MAX_DEPTH_BY_DEPTH_BITS, MAX_DEPTH_BY_KEY_BITS );

/**
 * @brief Returns depth of the given Morton key.
 */
inline
int MortonKeyGetDepth(KeyType k) {
    return k & DEPTH_MASK;
}

// Note this doesn't return a valid morton key when the incremented
// key overflows the overall region, but should be fine for GetBodyRange function
template <int DIM>
KeyType CalcMortonKeyNext(KeyType k) {
  int d = MortonKeyGetDepth(k);
  KeyType inc = (KeyType)1 << (DIM * (MAX_DEPTH - d) + DEPTH_BIT_WIDTH);
  return k + inc;
}

template <class T>
T __id(const T& t) {
    return t;
}

/**
 * @brief Returns the range of bodies from an array of T (body type) that belong to the cell specified by the given key. 
 * @tparam DIM Dimension
 * @tparam T Body type.
 * @tparam Iter Iterator type of the body array.
 * @tparam Functor Functor type that retrieves morton key from a body type value.
 * @return returns std::pair of (pos, len)
 */
template <int DIM, class T, class Iter, class Functor>
KeyPair GetBodyRange(const KeyType k, Iter beg, Iter end, Functor get_key = __id<T>) {
    // When used in Refine(), a cells has sometimes no body.
    // In this special case, just returns (0, 0)
    if (beg == end) return std::make_pair(0, 0);

    auto less_than = [get_key] (const T &hn, KeyType k) {
        return get_key(hn) < k;
    };

    auto fst = std::lower_bound(beg, end, k, less_than); // first node 
    auto lst = std::lower_bound(fst, end, CalcMortonKeyNext<DIM>(k), less_than); // last node

    assert(lst <= end);
    
    return std::make_pair(fst - beg, lst - fst); // returns (pos, nb)
}

/**
 * @brief std::vector version of GetBodyRange
 */
template<int DIM, class T, class Functor>
KeyPair GetBodyRange(const KeyType k, const std::vector<T> &hn, Functor get_key = __id<T>) {
    return GetBodyRange<DIM, T, typename std::vector<T>::const_iterator>(k, hn.begin(), hn.end(), get_key);
}

/**
 * @brief Set depth information in a Morton key.
 */
inline
KeyType MortonKeyAppendDepth(KeyType k, int depth) {
    k = (k << DEPTH_BIT_WIDTH) | depth;
    return k;
}

inline
KeyType MortonKeyRemoveDepth(KeyType k) {
    return k >> DEPTH_BIT_WIDTH;
}

inline
KeyType MortonKeyIncrementDepth(KeyType k, int inc) {
    int depth = MortonKeyGetDepth(k);
    depth += inc;
#ifdef TAPAS_DEBUG
    if (depth > MAX_DEPTH) {
        TAPAS_LOG_ERROR() << "Exceeded the maximum allowable depth: " << MAX_DEPTH << std::endl;
        TAPAS_DIE();
    }
#endif  
    k = MortonKeyRemoveDepth(k);
    return MortonKeyAppendDepth(k, depth);
}

template <int DIM>
bool MortonKeyIsDescendant(KeyType asc, KeyType dsc) {
    int depth = MortonKeyGetDepth(asc);
    if (depth >= MortonKeyGetDepth(dsc)) return false;
    int s = (MAX_DEPTH - depth) * DIM + DEPTH_BIT_WIDTH;
    asc >>= s;
    dsc >>= s;
    return asc == dsc;
}

template <int DIM>
KeyType MortonKeyClearDescendants(KeyType k) {
    int d = MortonKeyGetDepth(k);
    KeyType m = ~((((KeyType)1 << ((MAX_DEPTH - d) * DIM)) - 1) << DEPTH_BIT_WIDTH);
    return k & m;
}

template <int DIM>
KeyType MortonKeyParent(KeyType k) {
    int d = MortonKeyGetDepth(k);
    if (d == 0) return k;
    k = MortonKeyIncrementDepth(k, -1);
    return MortonKeyClearDescendants<DIM>(k);
}

template <int DIM>
KeyType MortonKeyFirstChild(KeyType k) {
#ifdef TAPAS_DEBUG
  KeyType t = MortonKeyRemoveDepth(k);
  t = t & ~(~((KeyType)0) << (DIM * (MAX_DEPTH - MortonKeyGetDepth(k))));
  assert(t == 0);
#endif
  assert(MortonKeyGetDepth(k) < MAX_DEPTH);
  return MortonKeyIncrementDepth(k, 1);
}


/**
 * @brief Converts a Morton key to a human-readable string format
 *
 * The format looks like "[0]-000-000-000-<0>". The first [0] is a reserved 1 bit for overflow.
 * "000-000-000" part is the key body. Each group corresponds to 1 tree level. The last "[0]" is the depth.
 */
template <int DIM>
std::string MortonKeyDecode(KeyType k) {
  std::stringstream ss;
  // get the overflow bit
  int overlow_bit = (k >> (DIM * MAX_DEPTH + DEPTH_BIT_WIDTH)) & 1;
  ss << "[" << overlow_bit << "]" << "-";

  // Get the key body
  for (int depth = 0; depth < MAX_DEPTH; depth++) {
    int mask = (1 << DIM) - 1;
    int level_d_key = (MortonKeyRemoveDepth(k) >> ((MAX_DEPTH - depth) * DIM)) & mask;
    for (int dim=DIM-1; dim >= 0; dim--) {
      ss << ((level_d_key >> dim) & 1);
    }
    ss << "-";
  }

  // Get depth
  ss << "<" << MortonKeyGetDepth(k) << ">";

  return ss.str();
}


/**
 * @brief Calculate a morton key of MAX_DEPTH level from the given anchor.
 * @tparam DIM Dimension.
 * @param anchor A Dim-dimensional index.
 */
template <int DIM>
KeyType CalcFinestMortonKey(const tapas::Vec<DIM, int> &anchor) {
  for (int d = DIM-1; d >= 0; --d) {
    assert(anchor[d] <= pow(2, MAX_DEPTH));
  }
  
  KeyType k = 0;
  int mask = 1 << (MAX_DEPTH - 1);
  for (int i = 0; i < MAX_DEPTH; ++i) {
    for (int d = DIM-1; d >= 0; --d) {
      k = (k << 1) | ((anchor[d] & mask) >> (MAX_DEPTH - i - 1));
    }
    mask >>= 1;
  }
  return MortonKeyAppendDepth(k, MAX_DEPTH);
}

#if 0
template <int DIM>
KeyType CalcFinestMortonKey(const tapas::Vec<DIM, int> &anchor) {
  KeyType k = 0;
  int mask = 1 << (MAX_DEPTH - 1);
  for (int i = 0; i < MAX_DEPTH; ++i) {
    for (int d = DIM-1; d >= 0; --d) {
      k = (k << 1) | ((anchor[d] & mask) >> (MAX_DEPTH - i - 1));
    }
    mask >>= 1;
  }
  return MortonKeyAppendDepth(k, MAX_DEPTH);
}
#endif

} // namespace morton_common
} // namespace tapas


#endif // TAPAS_MORTON_COMMON_
