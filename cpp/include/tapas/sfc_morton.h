#ifndef _SFC_MORTON_H_
#define _SFC_MORTON_H_

#include <cstdint>
#include <vector>
#include <algorithm>
#include <utility>
#include <list>
#include <unordered_map>

#ifdef CPP14
# define CONSTEXPR constexpr
#else
# define CONSTEXPR
#endif

namespace {

// define our own constexpr version of max/min/pow
/**
 * \brief max() function that can be used in compile time
 */
template<class T, class ... Args>
constexpr T smax(T a, Args ... args ) noexcept { return smax(a, smax(args...)); }

/**
 * \brief max() function that can be used in compile time
 */
template<class T>
constexpr T smax(T a, T b) noexcept { return a > b ? a : b; }

/**
 * \brief max() function that can be used in compile time
 */
template<class T>
constexpr T smax(T a) noexcept { return a; }

/**
 * \brief min() function that can be used in compile time
 */
template<class T, class ... Args>
constexpr T smin(T a, Args ... args ) noexcept { return smin(a, smin(args...)); }

/**
 * \brief min() function that can be used in compile time
 */
template<class T>
constexpr T smin(T a, T b) noexcept { return a > b ? a : b; }

/**
 * \brief min() function that can be used in compile time
 */
template<class T>
constexpr T smin(T a) noexcept { return a; }

/**
 * \brief pow() function that can be used in compile time
 */
template<class T>
constexpr T spow(T base, T exp) noexcept {
  return exp <= 0 ? 1
      :  exp == 1 ? base
      :  base * spow(base, exp-1);
}

template<class T>
T __id(T v) { return v; }

} // anon namespace


namespace tapas {
namespace sfc {

/*
Dim Bits Depth_bits Max_depth
  2   32          2         4
  2   32          3         8
  2   32          4        13
  2   32          5        13
  2   32          6        12
  2   32          7        12
  2   32          8        11
  2   32          9        11

  2   64          2         4
  2   64          3         8
  2   64          4        16
  2   64          5        29
  2   64          6        28
  2   64          7        28
  2   64          8        27
  2   64          9        27

  2  128          2         4
  2  128          3         8
  2  128          4        16
  2  128          5        32
  2  128          6        60
  2  128          7        60
  2  128          8        59
  2  128          9        59

  3   32          2         4
  3   32          3         8
  3   32          4         9
  3   32          5         8
  3   32          6         8
  3   32          7         8
  3   32          8         7
  3   32          9         7

  3   64          2         4
  3   64          3         8
  3   64          4        16
  3   64          5        19
  3   64          6        19
  3   64          7        18
  3   64          8        18
  3   64          9        18

  3  128          2         4
  3  128          3         8
  3  128          4        16
  3  128          5        32
  3  128          6        40
  3  128          7        40
  3  128          8        39
  3  128          9        39
 */

/**
 * A class implementing Morton key
 */
template<int _Dim, class _type>
class Morton {
 public:
  using KeyType = _type;
  
  typedef std::list<KeyType> KeyList;
  typedef std::vector<KeyType> KeyVector;
  typedef std::unordered_set<KeyType> KeySet;
  typedef std::pair<KeyType, KeyType> KeyPair;
  
  static const int Dim = _Dim;

  /**
   * Returns the best depth bits for the given Dim and KeyType
   */
  static constexpr int DepthBits() noexcept {
    return 0 ? 0
        : (Dim == 2 && sizeof(KeyType) ==  4) ? 4  // max depth = 13
        : (Dim == 2 && sizeof(KeyType) ==  8) ? 5  // max depth = 29
        : (Dim == 2 && sizeof(KeyType) == 16) ? 6  // max depth = 60
        : (Dim == 3 && sizeof(KeyType) ==  4) ? 4  // max depth =  9
        : (Dim == 3 && sizeof(KeyType) ==  8) ? 5  // max depth = 19
        : (Dim == 3 && sizeof(KeyType) == 16) ? 6  // max depth = 40
        : 0;
  }

  /**
   * Returns the maximum depth for the given Dim and KeyType
   */
  static constexpr int MaxDepth() noexcept {
    return 0 ? 0
        : (Dim == 2 && sizeof(KeyType) ==  4) ? 13 // max depth = 13
        : (Dim == 2 && sizeof(KeyType) ==  8) ? 29 // max depth = 29
        : (Dim == 2 && sizeof(KeyType) == 16) ? 60 // max depth = 60
        : (Dim == 3 && sizeof(KeyType) ==  4) ?  9 // max depth =  9
        : (Dim == 3 && sizeof(KeyType) ==  8) ? 19 // max depth = 19
        : (Dim == 3 && sizeof(KeyType) == 16) ? 40 // max depth = 40
        : 0;
  }

  static const constexpr int MAX_DEPTH = MaxDepth();
  static const constexpr int DEPTH_BIT_WIDTH = DepthBits();

  static constexpr KeyType kDepthMask = (1 << DepthBits()) - 1;

  static inline constexpr
  int GetDepth(KeyType k) noexcept {
    return k & kDepthMask;
  }

  /**
   * \brief Get the next key of k of the same depth in Morton order.
   */
  static inline CONSTEXPR
  KeyType GetNext(KeyType k) noexcept {
    int d = GetDepth(k);
    KeyType inc = (KeyType)1 << (Dim * (MaxDepth() - d) + DepthBits());
    return k + inc;
  }

  static inline CONSTEXPR
  KeyType AppendDepth(KeyType k, int depth) {
    return (k << DepthBits()) | depth;
  }

  static inline CONSTEXPR
  KeyType RemoveDepth(KeyType k) {
    return k >> DepthBits();
  }

  static inline CONSTEXPR
  KeyType IncrementDepth(KeyType k, int inc) {
    TAPAS_ASSERT(GetDepth(k) <= MAX_DEPTH);
    return AppendDepth(RemoveDepth(k),
                       GetDepth(k) + inc);
  }

  /**
   * Increment the depth of k by inc
   */
  static inline CONSTEXPR
  KeyType IncrDepth(KeyType k, int inc) {
    int d = GetDepth(k) + inc;
    
    TAPAS_ASSERT(0 <= d && d <= MaxDepth());
    
    return AppendDepth(RemoveDepth(k), d);
  }

  /**
   * \brief Returns if asc is an ancestor of dsc.
   */
  static inline CONSTEXPR
  bool IsDescendant(KeyType asc, KeyType dsc) {
    int depth = GetDepth(asc);
    if (depth >= GetDepth(dsc)) return false;
    int s = (MaxDepth() - depth) * Dim + DepthBits();
    asc >>= s;
    dsc >>= s;
    return asc == dsc;
  }

  static inline CONSTEXPR
  KeyType ClearDescendants(KeyType k) {
    int d = GetDepth(k);
    KeyType m = ~((((KeyType)1 << ((MaxDepth() - d) * Dim)) - 1) << DepthBits());
    return k & m;
  }

  static inline CONSTEXPR
  KeyType Parent(KeyType k) {
    int d = GetDepth(k);
    if (d == 0) return k;
    k = IncrementDepth(k, -1);
    return ClearDescendants(k);
  }

  static inline CONSTEXPR
  KeyType FirstChild(KeyType k) {
#ifdef TAPAS_DEBUG
    KeyType t = RemoveDepth(k);
    t = t & ~(~((KeyType)0) << (Dim * (MaxDepth() - GetDepth(k))));
    assert(t == 0);
#endif
    TAPAS_ASSERT(GetDepth(k) < MaxDepth());
    return IncrementDepth(k, 1);
  }

  static inline CONSTEXPR
  KeyType Child(KeyType k, int child_idx) {
    TAPAS_ASSERT(child_idx < (1 << Dim));
    k = IncrementDepth(k, 1);
    int d = GetDepth(k);
    return k | ((KeyType)child_idx << ((MaxDepth() - d) * Dim + DepthBits()));
  }

  static CONSTEXPR
  /**
   * @brief Returns a std::vector of parent's children
   */
  std::vector<KeyType> GetChildren(KeyType parent) {
    std::vector<KeyType> ret;
  
    KeyType child_key = FirstChild(parent);
    int nchild = (1 << Dim);
    
    for (int child_idx = 0; child_idx < nchild; child_idx++) {
      ret.push_back(child_key);
      child_key = GetNext(child_key);
    }

    return ret;
  }


  static inline CONSTEXPR
  KeyType FinestAncestor(KeyType x, KeyType y) {
    int min_depth = std::min(GetDepth(x),
                             GetDepth(y));
    x = RemoveDepth(x);
    y = RemoveDepth(y);
    KeyType a = ~(x ^ y);
    int common_depth = 0;
    for (; common_depth < min_depth; ++common_depth) {
      KeyType t = (a >> (MaxDepth() - common_depth -1) * Dim) & ((1 << Dim) - 1);
      if (t != ((1 << Dim) -1)) break;
    }
    int common_bit_len = common_depth * Dim;
    KeyType kOne = 1;
    KeyType mask = ((kOne << common_bit_len) - 1) << (MaxDepth() * Dim - common_bit_len);
    return AppendDepth(x & mask, common_depth);
  }

  static
  std::string Decode(KeyType k) {
    std::stringstream ss;
    // get the overflow bit
    int overlow_bit = (k >> (Dim * MaxDepth() + DepthBits())) & 1;
    ss << "[" << overlow_bit << "]" << "-";

    // Get the key body
    for (int depth = 0; depth < MaxDepth(); depth++) {
      int mask = (1 << Dim) - 1;
      int level_d_key = (RemoveDepth(k) >> ((MaxDepth() - depth) * Dim)) & mask;
      for (int dim = Dim-1; dim >= 0; dim--) {
        ss << ((level_d_key >> dim) & 1);
      }
      ss << "-";
    }

    // Get depth
    ss << "<" << GetDepth(k) << ">";

    return ss.str();
  }

  static
  KeyType CalcFinestKey(const tapas::Vec<Dim, int> &anchor) {
    for (int d = Dim-1; d >= 0; --d) {
      assert(anchor[d] <= pow(2, MaxDepth()));
    }
  
    KeyType k = 0;
    int mask = 1 << (MaxDepth() - 1);
    for (int i = 0; i < MaxDepth(); ++i) {
      for (int d = Dim-1; d >= 0; --d) {
        k = (k << 1) | ((anchor[d] & mask) >> (MaxDepth() - i - 1));
      }
      mask >>= 1;
    }
    return AppendDepth(k, MaxDepth());
  }

  template<class Iter>
  static void GetDescendantRange(KeyType k, Iter beg, Iter end, index_t &b, index_t &e) {
    auto comp = [](KeyType a, KeyType b) {
      return RemoveDepth(a) < RemoveDepth(b);
    };

    KeyType kn = GetNext(k);

    auto fst = std::lower_bound(beg, end, k, comp);
    auto lst = std::lower_bound(fst, end, kn, comp);
    b = fst - beg;
    e = lst - beg;
  }

  /**
   * @brief Find the range(beg,end) of cells/bodies that the cell
   */
  template<class T>
  static inline 
  void FindRangeByKey(const T& vec, KeyType k, index_t &range_beg, index_t &range_end) {
    auto beg = std::begin(vec);
    auto end = std::end(vec);
    range_beg = std::lower_bound(beg, end, k) - beg;
    range_end = std::lower_bound(beg, end, GetNext(k)) - beg;
  }

  /**
   * \brief returns a simplified, human-readable string representation of the key.
   * The returned string cannot be converted to the original key back.
   * NOTE: It might be useful to make this function constexpr, but 
   *       we use stringstream in the current implementation.
   * Example: "0461...7906"
   */
  static
  std::string Simplify(KeyType k) {
    int W=4;
    std::stringstream ss;
    ss << std::setw(20) << std::setfill('0') << k;
    std::string s = ss.str();
    s.replace(s.begin()+W, s.end()-W, "...");
    return s;
  }
};

}
}


#endif // _SFC_MORTON_H_
