#ifndef TAPAS_COMMON_H_
#define TAPAS_COMMON_H_

#if defined(__GNUG__) && !defined(__llvm__)
# define GCC_VERSION (__GNUC__ * 10000              \
                      + __GNUC_MINOR__ * 100        \
                      + __GNUC_PATCHLEVEL__)
# if GCC_VERSION < 40801
#  error "Tapas requires gcc/g++ >= 4.8.1"
# endif
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

//#define INLINE 
#define INLINE __attribute__((always_inline))

// for debug
#include <iomanip>
#include <unistd.h>
#include <sys/syscall.h> // for gettid()
#include <sys/types.h>   // for gettid()

#if defined(EXAFMM_TAPAS_MPI) || defined(USE_MPI) // FIXME: EXAFMM_TAPAS_MPI is only for debug
#include <mpi.h>
#endif

namespace {

#define DEBUG_WRITE

class Stderr {
  std::ostream *fs_;

 public:
  Stderr(const char *label) : fs_(nullptr) {
#ifdef DEBUG_WRITE
#if defined(EXAFMM_TAPAS_MPI) || defined(USE_MPI)  // FIXME: EXAFMM_TAPAS_MPI is only for debug
    pid_t tid = syscall(SYS_gettid);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    const char *rank="s";
    int tid=0;
#endif
    std::stringstream ss;
    ss << label << "."
       << rank  << "."
       << tid
       << ".stderr.txt";
    fs_ = new std::ofstream(ss.str().c_str(), std::ios_base::app);
#else
    fs_ = new std::stringstream();
#endif
  }
  
  ~Stderr() {
    assert(fs_ != nullptr);
    delete fs_;
    fs_ = nullptr;
  }

  std::ostream &out() {
    assert(fs_ != nullptr);
    return *fs_;
  }
};

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
 * @brief Holder of template parameter types.
 */
template<int _DIM, class _FP, class _BT, class _BT_ATTR, class _ATTR,
         class _Threading,
         class _SFC>
struct TapasStaticParams {
  static const int Dim = _DIM;  //!< dimension of simulation space
  typedef _FP FP;               //!< Floating point types
  typedef _BT BT;               //!< body info
  typedef _BT_ATTR BT_ATTR;     //!< body attributes
  typedef _ATTR ATTR;           //!< cell attributes
  typedef _Threading Threading; //!< threading policy
  typedef _SFC SFC;             //!< SFC implementation class
  
  // FIXME: the `SFC` class should not be here, because
  //        the concept of `SFC` is specific to HOT partitioning algorithm.
  //        As of now, HOT is the only implemented partitioning algorithm.
};


/**
 * @brief Sort vals using keys (assuming T1 is comparable). Both of keys and vals are sorted.
 *
 * @param keys keys
 * @param vals Values to be sorted.
 *
 */
template<class T1, class T2>
void SortByKeys(std::vector<T1> &keys, std::vector<T2> &vals) {
  assert(keys.size() == vals.size());
  
  auto len = keys.size();

  std::vector<size_t> perm(len); // Permutation

  for (size_t i = 0; i < len; i++) {
    perm[i] = i;
  }

  std::sort(std::begin(perm), std::end(perm),
            [&keys](size_t a, size_t b) { return keys[a] < keys[b]; });

  std::vector<T1> keys2(len); // sorted keys
  std::vector<T2> vals2(len); // sorted vals
  for (size_t i = 0; i < len; i++) {
    size_t idx = perm[i];
    vals2[i] = vals[idx];
    keys2[i] = keys[idx];
  }
  
  keys = keys2;
  vals = vals2;
}


} // namespace tapas

#endif /* TAPAS_COMMON_ */
