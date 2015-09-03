#ifndef TAPAS_MPI_UTIL_
#define TAPAS_MPI_UTIL_

#include <vector>
#include <algorithm>

#include <mpi.h>

namespace tapas {
namespace mpi {


// MPI-related utilities and wrappers
// TODO: wrap them as a pluggable policy/traits class
template<class T> struct MPI_DatatypeTraits {
  static MPI_Datatype type() {
    return MPI_BYTE;
  }
  static constexpr bool IsEmbType() {
    return false;
  }
};

#define DEF_MPI_DATATYPE(__ctype, __mpitype)        \
  template<> struct MPI_DatatypeTraits<__ctype>  {  \
    static MPI_Datatype type() {                    \
      return __mpitype;                             \
    }                                               \
    static constexpr bool IsEmbType() {           \
      return true;                                  \
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

/**
 * \brief Perform MPI_Alltoallv (version 1)
 * \tparam T data type
 * \param send_buf Data to be sent
 * \param dest Destination process number of each element of send_buf (i.e. send_buf[i] is sent to dest[i])
 * \param recv_buf (Output parameter) received data
 * \param src (Output parameter) source process number of each element of recv_buf (i.e. recv_buf[i] is from src[i])
 *
 * Caution: send_buf and dest will be sorted in-place.
 */
template<typename T>
void Alltoallv(std::vector<T>& send_buf, std::vector<int>& dest,
               std::vector<T>& recv_buf, std::vector<int>& src,
               MPI_Comm comm) {
  int mpi_size;

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  
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

  TAPAS_ASSERT(err == MPI_SUCCESS);

  std::vector<int> send_disp(mpi_size, 0); // displacement 
  std::vector<int> recv_disp(mpi_size, 0);
  
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
}

}
}

// MPI::COMPLEX Complex<float>
// MPI::DOUBLE_COMPLEX Complex<double>
// MPI::LONG_DOUBLE_COMPLEX Complex<long double>
// MPI::BYTE

#endif // TAPAS_MPI_UTIL_
