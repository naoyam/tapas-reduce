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
#include "tapas/single_node_hot.h"

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

typedef std::vector<float4> f4vec;

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
  if (c.IsLeaf()) {
    if (c.nb() == 0) {
      c.attr().w = 0.0;
#if 0
      c.attr().x = 0.0;
      c.attr().y = 0.0;
      c.attr().z = 0.0;
#endif
    } else if (c.nb() == 1) {
      c.attr() = c.body(0);
    }
    else {
      assert(false);
    }
  } else {
    //tapas::Map(approximate, c.subcells());
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
  if (!c1.IsLeaf()) {
    tapas::Map(interact, tapas::Product(c1.subcells(), c2), theta);
  } else if (c1.IsLeaf() && c1.nb() == 0) {
    return;
  } else if (c2.IsLeaf()) {
    if (c2.nb() == 0) {
      return;
    } else {
      // Both of c1 and c2 are leaves.
      // c1 and c2 have only one particle each. Calculate direct force.
      tapas::Map(ComputeForce, c1.bodies(), c2.body(0), EPS2);
    }
  } else {
    assert(c1.IsLeaf() && !c2.IsLeaf());
    
    // use apploximation
    const float4 &p1 = c1.body(0);
    real_t d = std::sqrt(distR2(c2.center(), p1));
    //real_t d = std::sqrt(distR2(c2.attr(), p1));
    real_t s = c2.width(0);

    if ((s/ d) < theta) {
      tapas::Map(ComputeForce, c1.bodies(), c2.attr(), EPS2);
    } else {
      tapas::Map(interact, tapas::Product(c1, c2.subcells()), theta);
    }
  }
}

typedef tapas::Vec<DIM, real_t> Vec3;

const char *dumpFileName = "approx.dat";

void DumpApproximate(Tapas::Cell *root) {
  // Dump random seed, number of bodies, and body_attrs
  // File name is fixed for now.
  std::ofstream out(dumpFileName);
  assert(out.good());
  out << seed << std::endl;
  out << N_total << std::endl;
  
  std::function<void (Tapas::Cell&)> dumper = [&out, &dumper](Tapas::Cell &cell) {
    out << cell.key() << " ";
    out << std::scientific << cell.attr().x << " ";
    out << std::scientific << cell.attr().y << " ";
    out << std::scientific << cell.attr().z << " ";
    out << std::scientific << cell.attr().w << " ";
    out << std::endl;
    if (!cell.IsLeaf()) {
      tapas::Map(dumper, cell.subcells());
    }
  };

  tapas::Map(dumper, *root);

  out.close();
}

void LoadApproximate(Tapas::Cell *root) {
#ifdef USE_MPI
  std::ifstream in(dumpFileName);
  assert(in.good());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int seed2 = 0, N_total2 = 0;
  in >> seed2 >> N_total2;
  assert(seed == seed2);
  assert(N_total == N_total2);

  auto &ht = root->data().ht_;

  while(!in.eof()) {
    Tapas::Cell::SFC::KeyType k;
    float4 v;
    in >> k >> v.x >> v.y >> v.z >> v.w;

    if (ht.count(k) > 0) {
      assert(ht.count(k) == 1);
      assert(ht[k] != nullptr);
      ht[k]->attr() = v;
    }
  }
#else
  (void)root; //hack: disable "unused parameter" warnings.
#endif
}

f4vec calc(f4vec &source) {
  Tapas::Region r(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0));
  Tapas::Cell *root = Tapas::Partition(source.data(), source.size(), r, 1);

  tapas::UpwardMap(approximate, *root); // or, simply: approximate(*root);
  
#if !defined(USE_MPI) && TAPAS_DEBUG
  DumpApproximate(root);
#endif
  
  real_t theta = 0.5;
  tapas::Map(interact, tapas::Product(*root, *root), theta);

  // Get the evaluation result from Tapas
  int nb = root->local_nb();
  f4vec out(&root->local_body_attr(0), &root->local_body_attr(0) + nb);
  source = f4vec(&root->local_body(0), &root->local_body(0) + nb);

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

void CheckResult(int np_check,
                 f4vec &sourceHost,
                 f4vec &targetTapas) {
  // sourceHost: local source bodies
  // targetTapas : target attrs computed by Tapas
  // tbodies : sampled target bodies (small portion of sourceHost)
  // tattrs  : target attrs to be computed directly in this function
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

#ifdef USE_MPI
  tapas::hot::BarrierExec([&tattrs](int, int) {
      std::stringstream ss;
      ss << "direct_" << mpi_size << ".dat";
      std::ofstream ofs(ss.str().c_str());
      for (size_t i = 0; i < tattrs.size(); i++) {
        ofs << tattrs[i].x << " "
            << tattrs[i].y << " "
            << tattrs[i].z << " "
            << tattrs[i].w << std::endl;
      }
      ofs.close();
    });
#else // ifdef USE_MPI
  std::stringstream ss;
  ss << "direct.dat";
  std::ofstream ofs(ss.str().c_str());
  for (size_t i = 0; i < tattrs.size(); i++) {
    ofs << tattrs[i].x << " "
        << tattrs[i].y << " "
        << tattrs[i].z << " "
        << tattrs[i].w << std::endl;
  }
  ofs.close();
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
  assert(N_total % mpi_size == 0);
  int N = N_total / mpi_size;
  OPS = 20. * N_total * N_total * 1e-9;  

  std::cout << "time n_total " << N_total << std::endl;
  std::cout << "time n_per_proc " << (N_total / mpi_size) << std::endl;

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
  std::cout << "time total_calc "   << std::scientific << toc-tic << " s" << std::endl;
  std::cout << "time total_gflops " << std::scientific << OPS / (toc-tic) << " GFlops" << std::endl;

  CheckResult(std::min(100, N), sourceHost, targetTapas);
  
  // DEALLOCATE
  
#ifdef USE_MPI
  MPI_Finalize();
#endif
}
