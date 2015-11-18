#ifndef TAPAS_MPI_UTIL_
#define TAPAS_MPI_UTIL_

#include <vector>
#include <algorithm>
#include <functional>

#include <mpi.h>

#include <tapas/common.h>

#include "tapas/debug_util.h"

namespace tapas {
namespace util {

/**
 * \brief Generic inclusive scan
 *
 * Prototypes of inclusive_scan and exclusive_scan are inspired by
 * "Working Draft, Technical Specification for C++ Extensions for Parallelism"
 * by Jared Hoberock (NVIDIA Corporation)
 * http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4310.html#parallel.alg.inclusive.scan
 */
template<class InputIterator, class OutputIterator,
         class BinaryOperation>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op) {
  using value_type = typename InputIterator::value_type;
  value_type tally = *first;
  InputIterator iter = first + 1;
  *result = tally;
  result++;

  while (iter != last) {
    tally = binary_op(tally, *iter);
    *result = tally;
    result++;
    iter++;
  }
  
  return result;
}

template<class InputIterator, class OutputIterator>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result) {
  return inclusive_scan(first, last, result,
                        std::plus<typename InputIterator::value_type>());
}


template<class InputIterator, class OutputIterator,
         class BinaryOperation>
OutputIterator
exclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result,
               BinaryOperation binary_op) {
  using value_type = typename InputIterator::value_type;
  value_type tally = value_type();
  InputIterator iter = first;
  
  *result = tally;
  result++;
  
  while (iter + 1 != last) {
    tally = binary_op(tally, *iter);
    *result = tally;
    result++;
    iter++;
  }
  
  return result;
}


template<class InputIterator, class OutputIterator>
OutputIterator
exclusive_scan(InputIterator first, InputIterator last,
               OutputIterator result) {
  return exclusive_scan(first, last, result,
                        std::plus<typename InputIterator::value_type>());
}



} // namespace tuil
} // namespace tapas


namespace tapas {
namespace mpi {

using tapas::util::exclusive_scan;

template<class T>
void* mpi_sendbuf_cast(const T* p) {
  return const_cast<void*>(reinterpret_cast<const void*>(p));
}

template<class T>
void* mpi_sendbuf_cast(T* p) {
  return reinterpret_cast<void*>(p);
}

// MPI-related utilities and wrappers
// TODO: wrap them as a pluggable policy/traits class
template<class T> struct MPI_DatatypeTraits {
  static constexpr MPI_Datatype type() {
    return MPI_BYTE;
  }
  static constexpr bool IsEmbType() {
    return false;
  }

  static constexpr int count(size_t n) {
    return sizeof(T) * n;
  }
};

#define DEF_MPI_DATATYPE(__ctype, __mpitype)        \
  template<> struct MPI_DatatypeTraits<__ctype>  {  \
    static MPI_Datatype type() {                    \
      return __mpitype;                             \
    }                                               \
    static constexpr bool IsEmbType() {             \
      return true;                                  \
    }                                               \
    static constexpr int count(size_t n) {          \
      return n;                                     \
    }                                               \
  };

DEF_MPI_DATATYPE(char, MPI_CHAR);
DEF_MPI_DATATYPE(unsigned char, MPI_UNSIGNED_CHAR);
DEF_MPI_DATATYPE(wchar_t, MPI_WCHAR);

DEF_MPI_DATATYPE(short, MPI_SHORT);
DEF_MPI_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT);

DEF_MPI_DATATYPE(int, MPI_INT);
DEF_MPI_DATATYPE(unsigned int, MPI_UNSIGNED);

DEF_MPI_DATATYPE(long, MPI_LONG);
DEF_MPI_DATATYPE(unsigned long, MPI_UNSIGNED_LONG);

DEF_MPI_DATATYPE(long long, MPI_LONG_LONG);
DEF_MPI_DATATYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG);

DEF_MPI_DATATYPE(float,  MPI_FLOAT);
DEF_MPI_DATATYPE(double, MPI_DOUBLE);
DEF_MPI_DATATYPE(long double, MPI_LONG_DOUBLE);

// MPI::COMPLEX Complex<float>
// MPI::DOUBLE_COMPLEX Complex<double>
// MPI::LONG_DOUBLE_COMPLEX Complex<long double>
// MPI::BYTE

template<typename T>
void Allreduce(const T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm) {
  auto kType = MPI_DatatypeTraits<T>::type();

  if (!MPI_DatatypeTraits<T>::IsEmbType()) {
    TAPAS_ASSERT(0 && "Allreduce() is not supported for user-defined types.");
  }

  int ret = MPI_Allreduce(mpi_sendbuf_cast(sendbuf), (void*)recvbuf, count, kType, op, comm);

  (void)ret; // to avoid warnings of 'unused variable'
  TAPAS_ASSERT(ret == MPI_SUCCESS);
}

template<typename T>
void Allreduce(const std::vector<T>& sendbuf, std::vector<T> &recvbuf,
               MPI_Op op, MPI_Comm comm) {
  size_t len = sendbuf.size();
  recvbuf.resize(len);

  Allreduce(sendbuf.data(), recvbuf.data(), (int) len, op, comm);
}

template<typename T>
void Allreduce(T sendval, T &recvval, MPI_Op op, MPI_Comm comm) {
  Allreduce(&sendval, &recvval, 1, op, comm);
}

template<typename T>
void Alltoall(const T *sendbuf, T *recvbuf, int count, MPI_Comm comm) {
  const auto kType = MPI_DatatypeTraits<T>::type();
  int size = MPI_DatatypeTraits<T>::IsEmbType() ? count : count * sizeof(T);
  int ret = ::MPI_Alltoall(sendbuf, size, kType,
                           recvbuf, size, kType,
                           comm);
  (void)ret; // to avoid warnings of 'unused variable'
  TAPAS_ASSERT(ret == MPI_SUCCESS);
}

template<typename T>
void Alltoall(const std::vector<T> &sendbuf, std::vector<T> &recvbuf, int count, MPI_Comm comm) {
  recvbuf.resize(sendbuf.size());
  Alltoall(sendbuf.data(), recvbuf.data(), count, comm);
}

/**
 * \brief Perform MPI_Alltoallv
 * \tparam T data type
 * \param send_buf Data to be sent
 * \param dest Destination process number of each element of send_buf (i.e. send_buf[i] is sent to dest[i])
 * \param recv_buf (Output parameter) received data
 * \param src (Output parameter) source process number of each element of recv_buf (i.e. recv_buf[i] is from src[i])
 *
 * Caution: send_buf and dest will be sorted in-place.
 */
template<typename T>
void Alltoallv2(std::vector<T>& send_buf, std::vector<int>& dest,
                std::vector<T>& recv_buf, std::vector<int>& src,
                MPI_Comm comm) {
  int mpi_size;

  MPI_Comm_size(comm, &mpi_size);
  
  TAPAS_ASSERT(send_buf.size() == dest.size());
  SortByKeys(dest, send_buf);

  std::vector<int> send_counts(mpi_size);
  for(int p = 0; p < mpi_size; p++) {
    auto range = std::equal_range(dest.begin(), dest.end(), p);
    send_counts[p] = range.second - range.first;
  }

  std::vector<int> recv_counts(mpi_size);

  int err = MPI_Alltoall((void*)send_counts.data(), 1, MPI_INT,
                         (void*)recv_counts.data(), 1, MPI_INT,
                         comm);

  if (err != MPI_SUCCESS) {
    TAPAS_ASSERT(!"MPI_Alltoall failed.");
  }

  std::vector<int> send_disp(mpi_size, 0); // displacement 
  std::vector<int> recv_disp(mpi_size, 0);

  // exclusive scan
  for (int p = 1; p < mpi_size; p++) {
    send_disp[p] = send_disp[p-1] + send_counts[p-1];

    recv_disp[p] = recv_disp[p-1] + recv_counts[p-1];
  }

  int total_recv_counts = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
  
  recv_buf.resize(total_recv_counts);

  auto kType = MPI_DatatypeTraits<T>::type();

  if(!MPI_DatatypeTraits<T>::IsEmbType()) {
    kType = MPI_BYTE;
    // If T is not an embedded type, MPI_BYTE and sizeof(T) is used.
    for (size_t i = 0; i < recv_disp.size(); i++) {
      recv_disp[i] *= sizeof(T);
      recv_counts[i] *= sizeof(T);
    }
    for (size_t i = 0; i < send_disp.size(); i++) {
      send_disp[i] *= sizeof(T);
      send_counts[i] *= sizeof(T);
    }
  }

  err = MPI_Alltoallv((void*)send_buf.data(), send_counts.data(), send_disp.data(), kType,
                      (void*)recv_buf.data(), recv_counts.data(), recv_disp.data(), kType,
                      comm);
  TAPAS_ASSERT(err == MPI_SUCCESS);
  
  // Build src[] array
  src.clear();
  src.resize(total_recv_counts, 0);
  
  int p = 0;
  for (int i = 0; i < total_recv_counts; i++) {
    while (p < mpi_size-1 && i >= recv_disp[p+1]) {
      p++;
    }
    src[i] = p;
  }

  auto src2 = src;
  src2.clear(); src2.resize(src.size(), 0);
  
  index_t pos = 0;
  for (size_t p = 0; p < recv_counts.size(); p++) {
    int num_recv_from_p = recv_counts[p];
    
    if(!MPI_DatatypeTraits<T>::IsEmbType()) {
      num_recv_from_p /= sizeof(T);
    }

    for (int i = 0; i < num_recv_from_p; i++, pos++) {
      src2[pos] = p;
    }
  }
  
#if 1
  // TODO: bug? May be src2 is the correct answer?
  src = src2;
#endif
}

template<class T>
void Alltoallv(const std::vector<T> &send_buf,
               const std::vector<int> &send_count,
               std::vector<T> &recv_buf, std::vector<int> &recv_count,
               MPI_Comm comm) {
  int mpi_size;
  
  MPI_Comm_size(comm, &mpi_size);

  recv_count.clear();
  recv_count.resize(mpi_size);

  int err = MPI_Alltoall((void*)send_count.data(), 1, MPI_INT,
                         (void*)recv_count.data(), 1, MPI_INT,
                         comm);

  if (err != MPI_SUCCESS) {
    TAPAS_ASSERT(!"MPI_Alltoall failed.");
  }

  std::vector<int> send_disp(mpi_size, 0); // displacement 
  std::vector<int> recv_disp(mpi_size, 0);

  // exclusive scan
  for (int p = 1; p < mpi_size; p++) {
    send_disp[p] = send_disp[p-1] + send_count[p-1];

    recv_disp[p] = recv_disp[p-1] + recv_count[p-1];
  }

  int total_recv_count = std::accumulate(recv_count.begin(), recv_count.end(), 0);
  
  recv_buf.resize(total_recv_count);

  auto kType = MPI_DatatypeTraits<T>::type();

  auto send_count2 = send_count;
  auto send_disp2 = send_disp;
  auto recv_count2 = recv_count;
  auto recv_disp2 = recv_disp;

  if(!MPI_DatatypeTraits<T>::IsEmbType()) {
    kType = MPI_BYTE;
    // If T is not an embedded type, MPI_BYTE and sizeof(T) is used.
    for (size_t i = 0; i < recv_disp.size(); i++) {
      recv_disp2[i] = recv_disp[i] * sizeof(T);
      recv_count2[i] = recv_count[i] * sizeof(T);
    }
    for (size_t i = 0; i < send_disp.size(); i++) {
      send_disp2[i] = send_disp[i] * sizeof(T);
      send_count2[i] = send_count[i] * sizeof(T);
    }
  }

  err = MPI_Alltoallv((void*)send_buf.data(), send_count2.data(), send_disp2.data(), kType,
                      (void*)recv_buf.data(), recv_count2.data(), recv_disp2.data(), kType,
                      comm);
  TAPAS_ASSERT(err == MPI_SUCCESS);
}

template<class T>
void Gather(const T& val, std::vector<T> &recvbuf, int root, MPI_Comm comm) {
  int size = -1;
  int rank = -1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  auto type = MPI_DatatypeTraits<T>::type();
  int count = MPI_DatatypeTraits<T>::count(1);
  
  if (rank == root) {
    recvbuf.clear();
    recvbuf.resize(size);
  } else {
    recvbuf.clear();
  }
  
  int ret = ::MPI_Gather(reinterpret_cast<const void*>(&val), count, type,
                         reinterpret_cast<void*>(&recvbuf[0]), count, type, root, comm);
  
  TAPAS_ASSERT(ret == MPI_SUCCESS); (void)ret;
}

template<class T>
void Gather(const std::vector<T> &sendbuf, std::vector<T> &recvbuf, int root, MPI_Comm comm) {
  int size = -1;
  int rank = -1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  auto type = MPI_DatatypeTraits<T>::type();
  int count = MPI_DatatypeTraits<T>::count(sendbuf.size());

  if (rank == root) {
    recvbuf.clear();
    recvbuf.resize(sendbuf.size() * size);
  } else {
    recvbuf.clear();
  }
  
  int ret = ::MPI_Gather(reinterpret_cast<const void*>(&sendbuf[0]), count, type,
                         reinterpret_cast<void*>(&recvbuf[0]), count, type, root, comm);
  
  TAPAS_ASSERT(ret == MPI_SUCCESS); (void)ret;
}

template<class T>
void Allgatherv(const std::vector<T> &sendbuf, std::vector<T> &recvbuf, MPI_Comm comm) {
  int size = -1, rank = -1;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
          
  int count = sendbuf.size();
  std::vector<int> recvcounts(size);
  
  auto kType = MPI_DatatypeTraits<T>::type();

  // Call allgather and create recvcount & displacements array.
  int ret = ::MPI_Allgather(mpi_sendbuf_cast(&count), 1, MPI_INT,
                            reinterpret_cast<void*>(recvcounts.data()), 1, MPI_INT, comm);
  (void)ret;
  TAPAS_ASSERT(ret == MPI_SUCCESS);
  
  std::vector<int> disp;
  exclusive_scan(recvcounts.begin(), recvcounts.end(), back_inserter(disp));

  int recvcount = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
  recvbuf.resize(recvcount);
  
  if (kType == MPI_BYTE) {
    count *= sizeof(T);
    recvcount *= sizeof(T);
    for (auto && c : recvcounts) c *= sizeof(T);
    for (auto && d : disp) d *= sizeof(T);
  }

  ret = ::MPI_Allgatherv(mpi_sendbuf_cast(sendbuf.data()), count, kType,
                         reinterpret_cast<void*>(recvbuf.data()), recvcounts.data(), disp.data(),
                         kType, comm);


  (void)ret;
  TAPAS_ASSERT(ret == MPI_SUCCESS);
}

template<class T>
void Bcast(std::vector<T> &buf, int root, MPI_Comm comm) {
  int rank = 0;

  MPI_Comm_rank(comm, &rank);

  ::MPI_Bcast(reinterpret_cast<void*>(buf.data()),
              MPI_DatatypeTraits<T>::count(buf.size()),
              MPI_DatatypeTraits<T>::type(),
              root,
              comm);
}


} // namespace mpi
} // namespace tapas

#endif // TAPAS_MPI_UTIL_
