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

#include <string>
#include <cstdlib>
#include <sstream>
#include <cassert>
#include <iostream>
#include <fstream>

#ifdef TAPAS_DEBUG
# if TAPAS_DEBUG == 0
#  undef TAPAS_DEBUG
# endif
#else  // TAPAS_DEBUG
# define TAPAS_DEBUG // default
#endif // TAPAS_DEBUG

// for debug
#include <iomanip>
#include <unistd.h>
#include <sys/syscall.h> // for gettid()
#include <sys/types.h>   // for gettid()

#ifdef EXAFMM_TAPAS_MPI
#include <mpi.h>
#endif

namespace {

#define DEBUG_WRITE

class Stderr {
  std::ostream *fs_;

 public:
  Stderr(const char *label) : fs_(nullptr) {
#ifdef DEBUG_WRITE
#ifdef EXAFMM_TAPAS_MPI
    pid_t tid = syscall(SYS_gettid);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    const char *rank="s";
    int tid=0;
#endif
    std::stringstream ss;
    ss << "stderr"
       << "." << rank
       << "." << tid
       << "." << label
       << ".txt";
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
    ss << v[i] << (i == v.size()-1 ? "" : " ");
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
 * @brief Holder of template parameter types (given by user code).
 */
template<int _DIM, class _FP, class _BT, class _BT_ATTR, class _ATTR, class _Threading>
struct TapasStaticParams {
  static const int Dim = _DIM; //!< dimension of simulation space
  typedef _FP FP;              //!< Floating point types
  typedef _BT BT;              //!< body info
  typedef _BT_ATTR BT_ATTR;    //!< body attributes
  typedef _ATTR ATTR;          //!< cell attributes
  typedef _Threading Threading; //!< threading policy
};

} // namespace tapas

#endif /* TAPAS_COMMON_ */
