#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <unistd.h>

#ifdef USE_MPI
# include <mpi.h>
#endif

#include "tapas.h"

//#define DIM (3)
typedef double real_t;

const constexpr int DIM = 3;
const real_t EPS2 = 1e-6;

// Global variables
int mpi_size = 1;
int mpi_rank = 0;
int seed = 0;     // random seed
int N_total = -1; // Total number of particles
real_t OPS = 0;

struct float4 {
  real_t x;
  real_t y;
  real_t z;
  real_t w;
};

float sum(const float f1, const float f2) {
  return f1 + f2;
}

typedef std::vector<float4> f4vec;

typedef tapas::BodyInfo<float4, 0> BodyInfo;

#ifdef USE_MPI
#include "tapas/hot.h"
#else
#include "tapas/single_node_hot.h"
#endif

#if 0
#if 0
typedef tapas::Tapas<DIM, real_t,
                     BodyInfo, // BT
                     float4,   // Cell attr
                     float4,   // body attr
                     tapas::HOT<DIM, tapas::sfc::Morton>,
                     tapas::threading::Default> Tapas;
#else
#include "tapas/single_node_hot.h"
typedef tapas::Tapas<DIM, real_t, BodyInfo,
                     float4,
                     float4,
                     tapas::SingleNodeHOT<DIM, tapas::sfc::Morton>,
                     tapas::threading::Default> Tapas;
#endif // 0
#endif // 0

// Select threading component: serial or MassiveThreads
#ifdef MTHREAD
#include "tapas/threading/massivethreads.h"
using Threading = tapas::threading::MassiveThreads;
#endif /* MTHREAD */

const constexpr size_t kBodyCoordOffset = 0;
struct BH_Params : public tapas::HOT<3, real_t, float4, kBodyCoordOffset, float4, float4> {
#ifdef MTHREAD
  using Threading = tapas::threading::MassiveThreads;
#endif
};

using TapasBH = tapas::Tapas<BH_Params>;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}


void P2P(f4vec &tattrs, f4vec &tbodies, f4vec &source, float eps2) {
  int ni = tbodies.size();
  int nj = source.size();
  
#pragma omp parallel for
  for (int i=0; i<ni; i++) {
    real_t ax = 0;
    real_t ay = 0;
    real_t az = 0;
    real_t phi = 0;
    real_t xi = tbodies[i].x;
    real_t yi = tbodies[i].y;
    real_t zi = tbodies[i].z;
    for (int j=0; j<nj; j++) {
      real_t dx = source[j].x - xi;
      real_t dy = source[j].y - yi;
      real_t dz = source[j].z - zi;
      real_t R2 = dx * dx + dy * dy + dz * dz + eps2;
      real_t invR = 1.0f / std::sqrt(R2);
      real_t invR3 = source[j].w * invR * invR * invR;
      phi += source[j].w * invR;
      ax += dx * invR3;
      ay += dy * invR3;
      az += dz * invR3;
    }
    tattrs[i].w += phi;
    tattrs[i].x += ax;
    tattrs[i].y += ay;
    tattrs[i].z += az;
  }
}

#if 0 // unsed. commented cout to supress 'unsed function' warnings.
static real_t distR2(const float4 &p, const float4 &q) {
  real_t dx = q.x - p.x;
  real_t dy = q.y - p.y;
  real_t dz = q.z - p.z;
  return dx * dx + dy * dy + dz * dz;
}
#endif

static real_t distR2(const tapas::Vec<3, double> &p, const float4 &q) {
  real_t dx = q.x - p[0];
  real_t dy = q.y - p[1];
  real_t dz = q.z - p[2];
  return dx * dx + dy * dy + dz * dz;
}

struct ComputeForce {
  
  template<class Body, class BodyAttr>
  inline void operator()(Body &p1, BodyAttr &p1_attr, float4 approx, real_t eps2) const {
    real_t dx = approx.x - p1.x; // const BodyType * BodyIterator::operator->()
    real_t dy = approx.y - p1.y;
    real_t dz = approx.z - p1.z;
    real_t R2 = dx * dx + dy * dy + dz * dz + eps2;
    real_t invR = 1.0 / std::sqrt(R2);
    real_t invR3 = invR * invR * invR;

    auto tmp = p1_attr;  // const ProxyBodyAttrType &BodyIterator::attr() const;
    tmp.x += dx * invR3 * approx.w;
    tmp.y += dy * invR3 * approx.w;
    tmp.z += dz * invR3 * approx.w;
    tmp.w += invR * approx.w;
    p1_attr = tmp; // ProxyBodyAttrType::operator=()
  }
};

struct Approximate {
  // Map acts on parent-child edges. Child can be the root, where the
  // parent parameter corresponds to a dummy cell that should not be
  // accessed 
  template<class Cell>
  inline void operator()(Cell &parent, Cell &child) {
    if (child.IsLeaf()) {
      if (child.nb() == 0) {
        auto attr = child.attr();
        attr.w = 0.0;
#if 0
        attr.x = 0.0;
        attr.y = 0.0;
        attr.z = 0.0;
#endif
        child.attr() = attr;
      } else if (child.nb() == 1) {
        child.attr() = child.body(0);
      }
      else {
        assert(false);
      }
    } else {
      TapasBH::Map(*this, child.subcells(), center);
      child.attr().x /= child.attr().w;
      child.attr().y /= child.attr().w;
      child.attr().z /= child.attr().w;
      if (!child.IsRoot()) {
        // Use "reduce" instead of "accumulate". In the C++ standard,
        // "reduce" is defined to be the out-of-order version of
        // "accumulate." See http://en.cppreference.com/w/cpp/algorithm/reduce
        Tapas::Reduce(parent.attr().w, child.attr().w, sum);
        Tapas::Reduce(parent.attr().x, child.attr().x, sum);
        Tapas::Reduce(parent.attr().y, child.attr().y, sum);
        Tapas::Reduce(parent.attr().z, child.attr().z, sum);
      }
    }
    return;
  }
};

struct interact {
  template<class Cell>
  inline void operator()(Cell &c1, Cell &c2, real_t theta) {
    if (!c1.IsLeaf()) {
      TapasBH::Map(*this, tapas::Product(c1.subcells(), c2), theta);
    } else if (c1.IsLeaf() && c1.nb() == 0) {
      return;
    } else if (c2.IsLeaf()) {
      if (c2.nb() == 0) {
        return;
      } else {
        // Both of c1 and c2 are leaves.
        // c1 and c2 have only one particle each. Calculate direct force.
        TapasBH::Map(ComputeForce(), c1.bodies(), c2.body(0), EPS2);
      }
    } else {
      assert(c1.IsLeaf() && !c2.IsLeaf());
      assert(c1.nb() == 1);
    
      // use apploximation
      const float4 &p1 = c1.body(0);
      real_t d = std::sqrt(distR2(c2.center(), p1));
      //real_t d = std::sqrt(distR2(c2.attr(), p1));
      real_t s = c2.width(0);
    
      if ((s/ d) < theta) {
        TapasBH::Map(ComputeForce(), c1.bodies(), c2.attr(), EPS2);
      } else {
        TapasBH::Map(*this, tapas::Product(c1, c2.subcells()), theta);
      }
    }
  }
};

typedef tapas::Vec<DIM, real_t> Vec3;

f4vec calc(f4vec &source) {
  TapasBH::Region r(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0));
  TapasBH::Cell *root = TapasBH::Partition(source.data(), source.size(), 1);

  TapasBH::Map(Approximate(), *root); // or, simply: approximate(*root);
  
  real_t theta = 0.5;
  TapasBH::Map(interact(), tapas::Product(*root, *root), theta);

  // Get the evaluation result from Tapas
  int nb = root->local_nb();
  f4vec out(&root->local_body_attr(0), &root->local_body_attr(0) + nb);
  source = f4vec(&root->local_body(0), &root->local_body(0) + nb);

  root->Report();

  return out;
}

void setRandSeed() {
  if (mpi_rank == 0) {
    if (getenv("TAPAS_SEED")) {
      seed = atoi(getenv("TAPAS_SEED"));
      std::cerr << "Seed = " << seed << std::endl;
    } else {
      seed = 0;
    }
  }

#ifdef USE_MPI
  MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  srand48(seed);
}

void parseOption(int *argc, char ***argv) {
  int result;
  int mpi_size = 1;
  int mpi_rank = 0;

#ifdef USE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
  
  while((result = getopt(*argc, *argv, "s:w:")) != -1) {
    switch(result) {
      case 'w':
        N_total = atoi(optarg) * mpi_size;
        break;
      case 's':
        N_total = atoi(optarg);
        break;
      case '?': 
        if (mpi_rank == 0) {
          std::cerr << "Usage:"
                    << "   $ " << (*argv)[0] << " -w N_per_proc -s N_total" << std::endl;
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        exit(0);
        break;
    }
  }
  
  *argc = optind;
}

/**
 * \brief Evaluate the accuracy of Tapas computation by comparing Tapas' result against direct N-body method.
 */
void CheckResult(int np_check,
                 f4vec &sourceHost,
                 f4vec &targetTapas) {
  // sourceHost: local source bodies
  // targetTapas : target attrs computed by Tapas
  // tbodies : sampled target bodies (small portion of sourceHost)
  // tattrs  : target attrs to be computed directly in this function

  TAPAS_ASSERT(np_check <= sourceHost.size());
  TAPAS_ASSERT(np_check <= targetTapas.size());
      
  f4vec tbodies(sourceHost.begin(), sourceHost.begin() + np_check);
  f4vec tattrs(np_check);
  
  // ------ Force evalution by direct computation (for validation)
  if (mpi_size == 1) {
    double tic = get_time();
    P2P(tattrs, tbodies, sourceHost, EPS2);
    double toc = get_time();
    std::cout << std::scientific << "No SSE : " << toc-tic << " s : "
              << OPS / (toc-tic) << " GFlops" << std::endl;
  }
#ifdef USE_MPI
  else {
    for (int i = 0; i < mpi_size; i++) {
      int src = i;
      int dst = (i + 1) % mpi_size;
      
      if (mpi_rank == src) {
        std::cerr << "Computing on rank " << mpi_rank << " and sending to " << dst << std::endl;
        P2P(tattrs, tbodies, sourceHost, EPS2);
        MPI_Send(tattrs.data(), tattrs.size() * sizeof(tattrs[0]), MPI_BYTE, dst, 0, MPI_COMM_WORLD);
        MPI_Send(tbodies.data(), tbodies.size() * sizeof(tbodies[0]), MPI_BYTE, dst, 0, MPI_COMM_WORLD);
      } else if (mpi_rank == dst) {
        MPI_Status st;
        MPI_Recv(tattrs.data(), tattrs.size() * sizeof(tattrs[0]), MPI_BYTE, src, 0, MPI_COMM_WORLD, &st);
        MPI_Recv(tbodies.data(), tbodies.size() * sizeof(tbodies[0]), MPI_BYTE, src, 0, MPI_COMM_WORLD, &st);
      }
      
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
#endif
  
#ifdef DUMP
  std::ofstream ref_out("bh_ref.txt");
  std::ofstream tapas_out("bh_tapas.txt");
#endif

  // COMPARE RESULTS
  if (mpi_rank == 0) {
    real_t pd = 0, pn = 0, fd = 0, fn = 0;
    
    for(int i = 0; i < np_check; i++ ) {
#ifdef DUMP
      ref_out << tattrs[i].x << " " << tattrs[i].y << " "
              << tattrs[i].z << " " << tattrs[i].w << std::endl;
      tapas_out << targetTapas[i].x << " " << targetTapas[i].y << " "
                << targetTapas[i].z << " " << targetTapas[i].w << std::endl;
#endif
      tattrs[i].w -= sourceHost[i].w / sqrtf(EPS2);
      targetTapas[i].w -= sourceHost[i].w / sqrtf(EPS2);
      pd += (tattrs[i].w - targetTapas[i].w) * (tattrs[i].w - targetTapas[i].w); // d^2, where d = potential diff
      pn += tattrs[i].w * tattrs[i].w;
      fd += (tattrs[i].x - targetTapas[i].x) * (tattrs[i].x - targetTapas[i].x)
            + (tattrs[i].y - targetTapas[i].y) * (tattrs[i].y - targetTapas[i].y)
            + (tattrs[i].z - targetTapas[i].z) * (tattrs[i].z - targetTapas[i].z);
      fn += tattrs[i].x * tattrs[i].x + tattrs[i].y * tattrs[i].y + tattrs[i].z * tattrs[i].z;
    }

    std::cout << std::scientific << "P ERR  : " << sqrtf(pd/pn) << std::endl;
    std::cout << std::scientific << "F ERR  : " << sqrtf(fd/fn) << std::endl;
  }
}

namespace {
template<class T>
inline T xmin(T t) {
  return t;
}

template<class T, class U, class ... Args>
inline T xmin(T t, U u, Args...args) {
  T m = (T) xmin(u, args...);
  return t > m ? m : t;
}

}

int main(int argc, char **argv) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  // print time and environment
  parseOption(&argc, &argv);

  if (N_total <= 0) {
    std::cerr << "Error: particle number is invalid or not specified." << std::endl;
    exit(-1);
  }
  
  // NOTE: Total number of particles is N * size (weak scaling)
  int N = N_total / mpi_size;
  OPS = 20. * N_total * N_total * 1e-9;

  f4vec sourceHost(N);

  setRandSeed();

  // initialze data
  for (int i = 0; i < N_total; i++) {
    double x = drand48();
    double y = drand48();
    double z = drand48();
    double w = drand48() / N_total;
    
    if (N * mpi_rank <= i && i < N * (mpi_rank+1)) {
      int j = i % N;
      sourceHost[j].x = x;
      sourceHost[j].y = y;
      sourceHost[j].z = z;
      sourceHost[j].w = w;
    }
  }

  if (mpi_rank == 0) {
    std::cout << std::scientific << "N      : " << N_total
              << " (" << N << " per proc)"
              << std::endl;
  }

  // ------ Force evalution by Tapas
  double tic = get_time();
  f4vec targetTapas = calc(sourceHost);
  double toc = get_time();
  
  tapas::debug::BarrierExec([=](int rank, int) {
      std::cout << "time " << rank << " total_calc "   << std::scientific << toc-tic << " s" << std::endl;
      std::cout << "time " << rank << " total_gflops " << std::scientific << OPS / (toc-tic) << " GFlops" << std::endl;
    });

  // check result
  // Compute interactions between sampled bodies and all others in all processes.
  // Determine how many bodies to sample
  int np_check = xmin(100, targetTapas.size(), sourceHost.size());

#ifdef USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &np_check, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif

  CheckResult(np_check, sourceHost, targetTapas);
  
  // DEALLOCATE

#ifdef USE_MPI
  MPI_Finalize();
#endif
  
  return 0;
}
