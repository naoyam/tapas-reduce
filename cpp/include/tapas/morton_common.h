#ifndef TAPAS_MORTON_COMMON_
#define TAPAS_MORTON_COMMON_

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
typedef int KeyType;

typedef std::list<KeyType> KeyList;
typedef std::vector<KeyType> KeyVector;
typedef std::unordered_set<KeyType> KeySet;
typedef std::pair<KeyType, KeyType> KeyPair;

using std::unordered_map;

const int DEPTH_BIT_WIDTH = 3;
const int DEPTH_MASK = (1 << DEPTH_BIT_WIDTH) - 1;
const int MAX_DEPTH = ((1 << DEPTH_BIT_WIDTH) - 1);

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
    KeyType inc = 1 << (DIM * (MAX_DEPTH - d) + DEPTH_BIT_WIDTH);
    return k + inc;
}

template <class T>
T id(const T& t) {
    return t;
}

/**
 * @brief Returns the range of bodies from an array of T (body type) that belong to the cell specified by the given key. 
 * @tparam DIM Dimension
 * @tparam T Body type.
 * @tparam Iter Iterator type of the body array.
 * @tparam Functor Functor type that retrieves morton key from a body type value.
 */
template <int DIM, class T, class Iter, class Functor>
KeyPair GetBodyRange(const KeyType k, Iter beg, Iter end, Functor get_key = id<T>) {
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
KeyPair GetBodyRange(const KeyType k, const std::vector<T> &hn, Functor get_key = id<T>) {
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
int MortonKeyIncrementDepth(KeyType k, int inc) {
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


} // namespace morton_common
} // namespace tapas


#endif // TAPAS_MORTON_COMMON_
