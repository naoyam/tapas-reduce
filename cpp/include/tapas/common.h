#ifndef TAPAS_COMMON_H_
#define TAPAS_COMMON_H_

#if defined(__clang__)
/* Clang/LLVM. ---------------------------------------------- */
# define TAPAS_COMPILER_CLANG

# define INLINE __attribute__((always_inline))

#elif defined(__ICC) || defined(__INTEL_COMPILER)
/* Intel ICC/ICPC. ------------------------------------------ */
# define TAPAS_COMPILER_INTEL

# define INLINE __forceinline

#elif defined(__GNUC__) || defined(__GNUG__)
/* GNU GCC/G++. --------------------------------------------- */
# define TAPAS_COPMILER_GCC
#  define GCC_VERSION (__GNUC__ * 10000             \
                      + __GNUC_MINOR__ * 100        \
                      + __GNUC_PATCHLEVEL__)
#  if GCC_VERSION < 40801
#    error "Tapas requires gcc/g++ >= 4.8.1"
#  endif

#define INLINE __attribute__((always_inline))

#endif

#if __cplusplus >= 201402L   // C++14
# define TAPAS_CPP14
#elif __cplusplus >= 201103L // C++11
# define TAPAS_CPP11
#else
# error "Tapas requires C++11 or later."
#endif

#include <cstdlib>
#include <cassert>

#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#ifdef TAPAS_DEBUG
# if TAPAS_DEBUG == 0
#  undef TAPAS_DEBUG
#  define TAPAS_MEASURE
# endif
#else  // TAPAS_DEBUG
# define TAPAS_DEBUG // default
#endif // TAPAS_DEBUG


// for debug
#include <iomanip>
#include <unistd.h>
#include <sys/syscall.h> // for gettid()
#include <sys/types.h>   // for gettid()

#if defined(EXAFMM_TAPAS_MPI) || defined(USE_MPI) // FIXME: EXAFMM_TAPAS_MPI is only for debug
#include <mpi.h>
#endif

namespace {

template<class T>
std::string join(const char *glue, const std::vector<T>& v) {
  std::stringstream ss;
  for (size_t i = 0; i < v.size(); i++) {
    ss << v[i] << (i == v.size()-1 ? "" : glue);
  }
  return ss.str();
}

}

/**
 * @brief Main namespae of Tapas
 */
namespace tapas {

using std::string;
using std::ostream;

typedef long index_t;

inline void Exit(int status, const char *file, const char *func, int line) {
  if (status) {
    std::cerr << "Exiting at " << file << "@" << func << "#" << line << std::endl;
  }
  std::exit(status);  
}

#ifdef TAPAS_DEBUG
#define TAPAS_ASSERT(c) assert(c)
#else
#define TAPAS_ASSERT(c) do {} while (0)
#endif

#define TAPAS_DIE() do {                        \
    Exit(1, __FILE__, __FUNCTION__, __LINE__);   \
  } while (0)

// Special class to indicate none 
class NONE {};

class StringJoin {
  const std::string sep_;
  bool first_;
  std::ostringstream ss_;
 public:
  StringJoin(const string &sep=", "): sep_(sep), first_(true) {}
  void append(const string &s) {
    if (!first_) ss_ << sep_;
    ss_ << s;
    first_ = false;
  }
        
  template <class T>
  std::ostringstream &operator<< (const T &s) {
    if (!first_) ss_ << sep_;
    ss_ << s;
    first_ = false;
    return ss_;
  }
  std::string get() const {
    return ss_.str();
  }
  std::string str() const {
    return get();
  }
};

/**
 * @brief Print a tapas::StringJoin object to a stream os
 */
inline std::ostream& operator<<(std::ostream &os,
                                const StringJoin &sj) {
  return os << sj.get();
}

/**
 * @brief Print keys of a container T to a stream os
 */
template <class T>
void PrintKeys(const T &s, std::ostream &os) {
    tapas::StringJoin sj;
    for (auto k: s) {
        sj << k;
    }
    os << "Key set: " << sj << std::endl;
}

/**
 * @brief Convert a type to an integer (not constexpr unfortunately)
 */
template<typename T>
struct Type2Int {
  static intptr_t value() {
    static size_t m = 0;
    return (intptr_t) &m;
  }
};

/**
 * @brief Sort vals using keys (assuming T1 is comparable). Both of keys and vals are sorted.
 *
 * @param keys keys
 * @param vals Values to be sorted.
 *
 */
template<class C1, class C2>
void SortByKeys(C1 &keys, C2 &vals) {
  assert(keys.size() == vals.size());
  
  auto len = keys.size();

  std::vector<size_t> perm(len); // Permutation

  for (size_t i = 0; i < len; i++) {
    perm[i] = i;
  }

  std::sort(std::begin(perm), std::end(perm),
            [&keys](size_t a, size_t b) { return keys[a] < keys[b]; });

  C1 keys2(len); // sorted keys
  C2 vals2(len); // sorted vals
  for (size_t i = 0; i < len; i++) {
    size_t idx = perm[i];
    vals2[i] = vals[idx];
    keys2[i] = keys[idx];
  }
  
  keys = keys2;
  vals = vals2;
}
} // namespace tapas

#ifdef __CUDACC__

#define CUDA_SAFE_CALL(expr)                                            \
  do {                                                                  \
    cudaError_t err = (expr);                                           \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "[Error] %s failed: %s (error code: %d) at %s:%d\n", \
              #expr, cudaGetErrorString(err), err, __FILE__, __LINE__); \
      exit(err);                                                        \
    }                                                                   \
  } while(0)

#endif
  
#endif /* TAPAS_COMMON_ */
