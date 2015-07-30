#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sys/time.h>

#ifdef USE_MPI
# include <mpi.h>
#endif

#include "tapas.h"

//#define DIM (3)
typedef double real_t;

const constexpr int DIM = 3;
const real_t EPS2 = 1e-6;
int mpi_size = 1;
int mpi_rank = 0;

struct float4 {
  real_t x;
  real_t y;
  real_t z;
  real_t w;
};

typedef tapas::BodyInfo<float4, 0> BodyInfo;

#ifdef USE_MPI
#include "tapas/hot.h"
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
#endif

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}


void P2P(float4 *target, float4 *source, int ni, int nj, float eps2) {
#pragma omp parallel for
  for (int i=0; i<ni; i++) {
    real_t ax = 0;
    real_t ay = 0;
    real_t az = 0;
    real_t phi = 0;
    real_t xi = source[i].x;
    real_t yi = source[i].y;
    real_t zi = source[i].z;
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
    target[i].w = phi;
    target[i].x = ax;
    target[i].y = ay;
    target[i].z = az;
  }
}

static real_t distR2(const float4 &p, const float4 &q) {
  real_t dx = q.x - p.x;
  real_t dy = q.y - p.y;
  real_t dz = q.z - p.z;
  return dx * dx + dy * dy + dz * dz;
}


static void ComputeForce(Tapas::BodyIterator &p1, 
                         float4 approx, real_t eps2) {
  real_t dx = approx.x - p1->x;
  real_t dy = approx.y - p1->y;
  real_t dz = approx.z - p1->z;
  real_t R2 = dx * dx + dy * dy + dz * dz + eps2;
  real_t invR = 1.0 / std::sqrt(R2);
  real_t invR3 = invR * invR * invR;
  p1.attr().x += dx * invR3 * approx.w;
  p1.attr().y += dy * invR3 * approx.w;
  p1.attr().z += dz * invR3 * approx.w;
  p1.attr().w += invR * approx.w;
}

static void approximate(Tapas::Cell &c) {
#ifdef DUMP
  std::cerr << "Approximate: " << c.key() << std::endl;
#endif

  if (c.nb() == 0) {
    c.attr().w = 0.0;
#if 0
    c.attr().x = 0.0;
    c.attr().y = 0.0;
    c.attr().z = 0.0;
#endif

#ifdef DUMP
    std::cerr << "Empty" << std::endl;
#endif

  } else if (c.nb() == 1) {
    c.attr() = c.body(0);
#ifdef DUMP
    std::cerr << "One particle" << std::endl;
#endif
  } else {
    tapas::Map(approximate, c.subcells());
    float4 center = {0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < c.nsubcells(); ++i) {
      Tapas::Cell &sc = c.subcell(i);
      center.w += sc.attr().w;
      center.x += sc.attr().x * sc.attr().w;
      center.y += sc.attr().y * sc.attr().w;
      center.z += sc.attr().z * sc.attr().w;
    }
    center.x /= center.w;
    center.y /= center.w;
    center.z /= center.w;
    c.attr() = center;
  }
}

static void interact(Tapas::Cell &c1, Tapas::Cell &c2, real_t theta) {
#ifdef TAPAS_DEBUG
  {
    Stderr e("interact");
    e.out() << "Head." << std::endl;
    e.out() << "\t" << Tapas::Cell::SFC::Decode(c1.key()) << std::endl;
    e.out() << "\t" << Tapas::Cell::SFC::Decode(c2.key()) << std::endl;
  }
#endif
  
  if (c1.nb() == 0 || c2.nb() == 0) {
    return;
  } else if (!c1.IsLeaf()) {
    tapas::Map(interact, tapas::Product(c1.subcells(), c2), theta);
  } else if (c2.IsLeaf()) {
#ifdef TAPAS_DEBUG
    {
      Stderr e("interact");
      e.out() << "Leaf/leaf." << std::endl;
      e.out() << "\t" << Tapas::Cell::SFC::Decode(c1.key()) << std::endl;
      e.out() << "\t" << Tapas::Cell::SFC::Decode(c2.key()) << std::endl;
    }
#endif
    
    // c1 and c2 have only one particle each. Calculate direct force.
    //tapas::Map(ComputeForce, tapas::Product(c1.particles(),
    //c2.particles()));
    tapas::Map(ComputeForce, c1.bodies(), c2.body(0), EPS2);
  } else {
    // use apploximation
    const float4 &p1 = c1.body(0);
    real_t d = std::sqrt(distR2(c2.attr(), p1));
    real_t s = c2.width(0);
    
    if ((s/ d) < theta) {
#ifdef TAPAS_DEBUG
      {
        Stderr e("interact");
        e.out() << "Leaf/branch. far enough. approximate." << std::endl;
        e.out() << "\t" << Tapas::Cell::SFC::Decode(c1.key()) << std::endl;
        e.out() << "\t" << Tapas::Cell::SFC::Decode(c2.key()) << std::endl;
      }
#endif
      tapas::Map(ComputeForce, c1.bodies(), c2.attr(), EPS2);
    } else {
#ifdef TAPAS_DEBUG
      {
        Stderr e("interact");
        e.out() << "Leaf/branch. close. recursive." << std::endl;
        e.out() << "\t" << Tapas::Cell::SFC::Decode(c1.key()) << std::endl;
        e.out() << "\t" << Tapas::Cell::SFC::Decode(c2.key()) << std::endl;
      }
#endif
      tapas::Map(interact, tapas::Product(c1, c2.subcells()), theta);
    }
  }
}

typedef tapas::Vec<DIM, real_t> Vec3;

float4 *calc(float4 *p, size_t np) {
#ifdef USE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
  
  // PartitionBSP is a function that partitions the given set of
  // particles by the binary space partitioning. The result is a
  // octree for 3D particles and a quadtree for 2D particles.
  Tapas::Region r(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0));
  Tapas::Cell *root = Tapas::Partition(p, np, r, 1);


  // FIXME: this line is skipped for multi-process run because
  //        ExchangeLET for approximate phase is not yet implemented.
  if (mpi_size == 1) {
    tapas::Map(approximate, *root); // or, simply: approximate(*root);
  } else {
    // Load upward result from a file
  }
  
  real_t theta = 0.5;
  tapas::Map(interact, tapas::Product(*root, *root), theta);

  // Get the evaluation result from Tapas
  float4 *out = root->body_attrs();

  // Get the re-ordered sourceHost
  assert(np == root->nbodies());
  
  for (int i = 0; i < np; i++) {
    p[i] = root->body(i);
  }
  return out;
}

void setRandSeed() {
  int seed = 0;
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

// Total number of particles.
int N_total = -1;

void parseOption(int *argc, char ***argv) {
  int result;
  int mpi_size = 1;

#ifdef USE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
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
        std::cerr << "Usage:"
                  << "   $ " << (*argv)[0] << " -w N_per_proc -s N_total" << std::endl;
        exit(0);
        break;
    }
  }

  *argc = optind;
}

int main(int argc, char **argv) {
#ifdef USE_MPI
  int provided, required = MPI_THREAD_MULTIPLE;
  MPI_Init_thread(&argc, &argv, required, &provided);
  assert(provided >= required);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  parseOption(&argc, &argv);

  if (N_total <= 0) {
    std::cerr << "Error: particle number is not specified." << std::endl;
    exit(-1);
  }

  // NOTE: Total number of particles is N * size (weak scaling)
  const real_t OPS = 20. * N_total * N_total * 1e-9;
  assert(N_total % mpi_size == 0);
  int N = N_total / mpi_size;

  std::cout << "time n_total " << N_total << std::endl;
  std::cout << "time n_per_proc " << (N_total / mpi_size) << std::endl;
  
  float4 *sourceHost = new float4 [N];
  float4 *targetHost = new float4 [N];

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
  float4 *targetTapas = calc(sourceHost, N);
  double toc = get_time();
  std::cout << "time total_calc "   << std::scientific << toc-tic << " s" << std::endl;
  std::cout << "time total_gflops " << std::scientific << OPS / (toc-tic) << " GFlops" << std::endl;

  // ------ Force evalution by direct computation (for validation)
  if (mpi_size == 1) {
    double tic = get_time();
    P2P(targetHost, sourceHost, N, N, EPS2);
    double toc = get_time();
    std::cout << std::scientific << "No SSE : " << toc-tic << " s : "
              << OPS / (toc-tic) << " GFlops" << std::endl;
  }
  
#ifdef DUMP
  std::ofstream ref_out("bh_ref.txt");
  std::ofstream tapas_out("bh_tapas.txt");
#endif

  // COMPARE RESULTS
  if (mpi_size == 1) {
    real_t pd = 0, pn = 0, fd = 0, fn = 0;
    for( int i=0; i<N; i++ ) {
#ifdef DUMP
      ref_out << targetHost[i].x << " " << targetHost[i].y << " "
              << targetHost[i].z << " " << targetHost[i].w << std::endl;
      tapas_out << targetTapas[i].x << " " << targetTapas[i].y << " "
                << targetTapas[i].z << " " << targetTapas[i].w << std::endl;
#endif
      targetHost[i].w -= sourceHost[i].w / sqrtf(EPS2);
      targetTapas[i].w -= sourceHost[i].w / sqrtf(EPS2);
      pd += (targetHost[i].w - targetTapas[i].w) * (targetHost[i].w - targetTapas[i].w);
      pn += targetHost[i].w * targetHost[i].w;
      fd += (targetHost[i].x - targetTapas[i].x) * (targetHost[i].x - targetTapas[i].x)
            + (targetHost[i].y - targetTapas[i].y) * (targetHost[i].y - targetTapas[i].y)
            + (targetHost[i].z - targetTapas[i].z) * (targetHost[i].z - targetTapas[i].z);
      fn += targetHost[i].x * targetHost[i].x + targetHost[i].y * targetHost[i].y + targetHost[i].z * targetHost[i].z;
    }
    std::cout << std::scientific << "P ERR  : " << sqrtf(pd/pn) << std::endl;
    std::cout << std::scientific << "F ERR  : " << sqrtf(fd/fn) << std::endl;
  } else {
    std::cout << "Skipping result check" << std::endl;
  }

// DEALLOCATE
  delete[] sourceHost;
  delete[] targetHost;
  //delete[] targetTapas;

#ifdef USE_MPI
  MPI_Finalize();
#endif
}
