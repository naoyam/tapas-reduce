#include <iostream>
#include <iomanip>

#include "tapas_exafmm.h"
#include "tapas/debug_util.h" // for t::d::DebugStream

#define ODDEVEN(n) ((((n) & 1) == 1) ? -1 : 1)
#define IPOW2N(n) ((n >= 0) ? 1 : ODDEVEN(n))

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <fstream>
#include <iomanip>

#ifdef USE_MPI
#include <mpi.h>
#endif

extern uint64_t numM2L;
extern uint64_t numP2P;

#ifdef COUNT
# define INC_M2L do { numM2L++; } while(0)
# else
# define INC_M2L
#endif

namespace {

const complex_t I(0.,1.);                                       // Imaginary
                                                                // unit
template <int DIM, class FP> inline
vec<DIM, FP> tovec(const tapas::Vec<DIM, FP> &src) {
  vec<DIM, FP> dst;
  for (int i = 0; i < DIM; ++i) {
    dst[i] = src[i];
  }
  return dst;
}

//! Get r,theta,phi from x,y,z
void cart2sph(real_t & r, real_t & theta, real_t & phi, vec3 dX) {
  r = sqrt(norm(dX));                                           // r = sqrt(x^2 + y^2 + z^2)
  theta = r == 0 ? 0 : acos(dX[2] / r);                         // theta = acos(z / r)
  phi = atan2(dX[1], dX[0]);                                    // phi = atan(y / x)
}

//! Spherical to cartesian coordinates
template<typename T>
void sph2cart(real_t r, real_t theta, real_t phi, T spherical, T & cartesian) {
  cartesian[0] = sin(theta) * cos(phi) * spherical[0]           // x component (not x itself)
    + cos(theta) * cos(phi) / r * spherical[1]
    - sin(phi) / r / sin(theta) * spherical[2];
  cartesian[1] = sin(theta) * sin(phi) * spherical[0]           // y component (not y itself)
    + cos(theta) * sin(phi) / r * spherical[1]
    + cos(phi) / r / sin(theta) * spherical[2];
  cartesian[2] = cos(theta) * spherical[0]                      // z component (not z itself)
    - sin(theta) / r * spherical[1];
}

//! Evaluate solid harmonics \f$ r^n Y_{n}^{m} \f$
void evalMultipole(real_t rho, real_t alpha, real_t beta, complex_t * Ynm, complex_t * YnmTheta) {
  real_t x = std::cos(alpha);                                   // x = cos(alpha)
  real_t y = std::sin(alpha);                                   // y = sin(alpha)
  real_t fact = 1;                                              // Initialize 2 * m + 1
  real_t pn = 1;                                                // Initialize Legendre polynomial Pn
  real_t rhom = 1;                                              // Initialize rho^m
  complex_t ei = std::exp(I * beta);                            // exp(i * beta)
  complex_t eim = 1.0;                                          // Initialize exp(i * m * beta)
  for (int m=0; m<P; m++) {                                     // Loop over m in Ynm
    real_t p = pn;                                              //  Associated Legendre polynomial Pnm
    int npn = m * m + 2 * m;                                    //  Index of Ynm for m > 0
    int nmn = m * m;                                            //  Index of Ynm for m < 0
    Ynm[npn] = rhom * p * eim;                                  //  rho^m * Ynm for m > 0
    Ynm[nmn] = std::conj(Ynm[npn]);                             //  Use conjugate relation for m < 0
    real_t p1 = p;                                              //  Pnm-1
    p = x * (2 * m + 1) * p1;                                   //  Pnm using recurrence relation
    YnmTheta[npn] = rhom * (p - (m + 1) * x * p1) / y * eim;    //  theta derivative of r^n * Ynm
    rhom *= rho;                                                //  rho^m
    real_t rhon = rhom;                                         //  rho^n
    for (int n=m+1; n<P; n++) {                                 //  Loop over n in Ynm
      int npm = n * n + n + m;                                  //   Index of Ynm for m > 0
      int nmm = n * n + n - m;                                  //   Index of Ynm for m < 0
      rhon /= -(n + m);                                         //   Update factorial
      Ynm[npm] = rhon * p * eim;                                //   rho^n * Ynm
      Ynm[nmm] = std::conj(Ynm[npm]);                           //   Use conjugate relation for m < 0
      real_t p2 = p1;                                           //   Pnm-2
      p1 = p;                                                   //   Pnm-1
      p = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);  //   Pnm using recurrence relation
      YnmTheta[npm] = rhon * ((n - m + 1) * p - (n + 1) * x * p1) / y * eim;// theta derivative
      rhon *= rho;                                              //   Update rho^n
    }                                                           //  End loop over n in Ynm
    rhom /= -(2 * m + 2) * (2 * m + 1);                         //  Update factorial
    pn = -pn * fact * y;                                        //  Pn
    fact += 2;                                                  //  2 * m + 1
    eim *= ei;                                                  //  Update exp(i * m * beta)
  }                                                             // End loop over m in Ynm
}

//! Evaluate singular harmonics \f$ r^{-n-1} Y_n^m \f$
void evalLocal(real_t rho, real_t alpha, real_t beta, complex_t * Ynm) {
  real_t x = std::cos(alpha);                                   // x = cos(alpha)
  real_t y = std::sin(alpha);                                   // y = sin(alpha)
  real_t fact = 1;                                              // Initialize 2 * m + 1
  real_t pn = 1;                                                // Initialize Legendre polynomial Pn
  real_t invR = -1.0 / rho;                                     // - 1 / rho
  real_t rhom = -invR;                                          // Initialize rho^(-m-1)
  complex_t ei = std::exp(I * beta);                            // exp(i * beta)
  complex_t eim = 1.0;                                          // Initialize exp(i * m * beta)
  for (int m=0; m<P; m++) {                                     // Loop over m in Ynm
    real_t p = pn;                                              //  Associated Legendre polynomial Pnm
    int npn = m * m + 2 * m;                                    //  Index of Ynm for m > 0
    int nmn = m * m;                                            //  Index of Ynm for m < 0
    Ynm[npn] = rhom * p * eim;                                  //  rho^(-m-1) * Ynm for m > 0
    Ynm[nmn] = std::conj(Ynm[npn]);                             //  Use conjugate relation for m < 0
    real_t p1 = p;                                              //  Pnm-1
    p = x * (2 * m + 1) * p1;                                   //  Pnm using recurrence relation
    rhom *= invR;                                               //  rho^(-m-1)
    real_t rhon = rhom;                                         //  rho^(-n-1)
    for (int n=m+1; n<P; n++) {                                 //  Loop over n in Ynm
      int npm = n * n + n + m;                                  //   Index of Ynm for m > 0
      int nmm = n * n + n - m;                                  //   Index of Ynm for m < 0
      Ynm[npm] = rhon * p * eim;                                //   rho^n * Ynm for m > 0
      Ynm[nmm] = std::conj(Ynm[npm]);                           //   Use conjugate relation for m < 0
      real_t p2 = p1;                                           //   Pnm-2
      p1 = p;                                                   //   Pnm-1
      p = (x * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);  //   Pnm using recurrence relation
      rhon *= invR * (n - m + 1);                               //   rho^(-n-1)
    }                                                           //  End loop over n in Ynm
    pn = -pn * fact * y;                                        //  Pn
    fact += 2;                                                  //  2 * m + 1
    eim *= ei;                                                  //  Update exp(i * m * beta)
  }                                                             // End loop over m in Ynm
}


template<class Cell>
void P2M(Cell &C) {
  complex_t Ynm[P*P], YnmTheta[P*P];

  auto attr = C.attr();
  
  for (size_t i = 0; i < C.nb(); ++i) {
    const Body &B = C.body(i);
    vec3 dX = B.X - tovec(C.center());
    real_t rho, alpha, beta;
    cart2sph(rho, alpha, beta, dX);
    evalMultipole(rho, alpha, beta, Ynm, YnmTheta);
    
    //e.out() << std::setw(10) << Tapas::SFC::Simplify(C.key()) << " B[" << i << "].SRC=" << B.SRC << std::endl;
    
    for (int n=0; n<P; n++) {
      for (int m=0; m<=n; m++) {
        int nm  = n * n + n - m;
        int nms = n * (n + 1) / 2 + m;
        tapas::Accumulate(attr.M[nms], B.SRC * Ynm[nm]);
      }
    }
  }
  C.attr() = attr;
  //e.out() << std::setw(10) << Tapas::SFC::Simplify(C.key()) << "M=" << C.attr().M << std::endl;
}

template<class Cell>
void M2M(Cell &C) {
  complex_t Ynm[P*P], YnmTheta[P*P];
  
  auto attr = C.attr();

  for (int i = 0; i < C.nsubcells(); ++i) {
    Cell &Cj=C.subcell(i);
    
    // Skip empty cell
    // NOTE: This is not allowed in
    // TODO: Do we want to allow this?
    //if (Cj.nb() == 0) continue;
    
    vec3 dX = tovec(C.center() - Cj.center());
    
    real_t rho, alpha, beta;
    cart2sph(rho, alpha, beta, dX);
    evalMultipole(rho, alpha, beta, Ynm, YnmTheta);

    for (int j=0; j<P; j++) {
      for (int k=0; k<=j; k++) {
        int jks = j * (j + 1) / 2 + k;
        complex_t M = 0;
        for (int n=0; n<=j; n++) {
          for (int m=std::max(-n,-j+k+n); m<=std::min(k-1,n); m++) {
            int jnkms = (j - n) * (j - n + 1) / 2 + k - m;
            int nm    = n * n + n - m;
            M += Cj.attr().M[jnkms] * Ynm[nm] * real_t(IPOW2N(m) * ODDEVEN(n));
          }
          for (int m=k; m<=std::min(n,j+k-n); m++) {
            int jnkms = (j - n) * (j - n + 1) / 2 - k + m;
            int nm    = n * n + n - m;
            M += std::conj(Cj.attr().M[jnkms]) * Ynm[nm] * real_t(ODDEVEN(k+n+m));
          }
        }
        attr.M[jks] += M;
      }
    }
  }
  C.attr() = attr;
}

template<class Cell>
void M2L(Cell &Ci, Cell &Cj, vec3 Xperiodic, bool mutual) {
  SCOREP_USER_REGION("M2L", SCOREP_USER_REGION_TYPE_FUNCTION);
  INC_M2L;
  
  complex_t Ynmi[P*P], Ynmj[P*P];
  //vec3 dX = Ci.attr().X - Cj.attr().X - Xperiodic;
  auto attr_i = Ci.attr();
  auto attr_j = Cj.attr();

  vec3 dX;
  asn(dX, Ci.center() - Cj.center());
  dX -= Xperiodic;

  real_t rho, alpha, beta;
  cart2sph(rho, alpha, beta, dX);
  evalLocal(rho, alpha, beta, Ynmi);
  if (mutual) evalLocal(rho, alpha+M_PI, beta, Ynmj);

  for (int j=0; j<P; j++) {
#if MASS
    real_t Cnm = std::real(Ci->M[0] * Cj->M[0]) * ODDEVEN(j);
#else
    real_t Cnm = ODDEVEN(j);
#endif

    for (int k=0; k<=j; k++) {
      int jks = j * (j + 1) / 2 + k;
      complex_t Li = 0, Lj = 0;
#if MASS
      int jk = j * j + j - k;
      Li += Cnm * Ynmi[jk];
      if (mutual) Lj += Cnm * Ynmj[jk];
      for (int n=1; n<P-j; n++)
#else
      for (int n=0; n<P-j; n++)
#endif
      {
        for (int m=-n; m<0; m++) {
          int nms  = n * (n + 1) / 2 - m;
          int jnkm = (j + n) * (j + n) + j + n + m - k;
          Li += std::conj(attr_j.M[nms]) * Cnm * Ynmi[jnkm];
          //std::cerr << "M: " << attr_j.M[nms] << std::conj(attr_j.M[nms]) << std::endl;
          //std::cerr << "Y: " << Ynmi[jnkm] << std::endl;
          if (mutual) Lj += std::conj(attr_i.M[nms]) * Cnm * Ynmj[jnkm];
        }
        for (int m=0; m<=n; m++) {
          int nms  = n * (n + 1) / 2 + m;
          int jnkm = (j + n) * (j + n) + j + n + m - k;
          real_t Cnm2 = Cnm * ODDEVEN((k-m)*(k<m)+m);
          Li += attr_j.M[nms] * Cnm2 * Ynmi[jnkm];
          //std::cerr << "M: " << attr_j.M[nms] << std::conj(attr_j.M[nms]) << std::endl;
          if (mutual) Lj += attr_i.M[nms] * Cnm2 * Ynmj[jnkm];
        }
      }

      // TODO: attr_j can be put out of the outer `for' loop
      attr_i.L[jks] += Li;
      if (mutual) {
        attr_j.L[jks] += Lj;
      }
    }
  }
  Ci.attr() = attr_i;
  if (mutual) Cj.attr() = attr_j;
}

void L2P(TapasFMM::Cell &c, Body &b, BodyAttr &ba) { // c is a pointer here to avoid NVCC's bug of parsing C++ code.
  complex_t Ynm[P*P], YnmTheta[P*P];
  vec3 dX = b.X - tovec(c.center());
  vec3 spherical = 0;
  vec3 cartesian = 0;
  real_t r, theta, phi;
  cart2sph(r, theta, phi, dX);
  evalMultipole(r, theta, phi, Ynm, YnmTheta);
  ba /= b.SRC;
  for (int n=0; n<P; n++) {
    int nm  = n * n + n;
    int nms = n * (n + 1) / 2;
    //b.TRG[0] += std::real(c->attr().L[nms] * Ynm[nm]);
    ba[0] += std::real(c.attr().L[nms] * Ynm[nm]);
    spherical[0] += std::real(c.attr().L[nms] * Ynm[nm]) / r * n;
    spherical[1] += std::real(c.attr().L[nms] * YnmTheta[nm]);
    for( int m=1; m<=n; m++) {
      nm  = n * n + n + m;
      nms = n * (n + 1) / 2 + m;
      //b.TRG[0] += 2 * std::real(c.attr().L[nms] * Ynm[nm]);
      ba[0] += 2 * std::real(c.attr().L[nms] * Ynm[nm]);
      spherical[0] += 2 * std::real(c.attr().L[nms] * Ynm[nm]) / r * n;
      spherical[1] += 2 * std::real(c.attr().L[nms] * YnmTheta[nm]);
      spherical[2] += 2 * std::real(c.attr().L[nms] * Ynm[nm] * I) * m;
    }
  }
  sph2cart(r, theta, phi, spherical, cartesian);
  ba[1] += cartesian[0];
  //b.TRG[2] += cartesian[1];
  ba[2] += cartesian[1];
  //b.TRG[3] += cartesian[2];
  ba[3] += cartesian[2]; 
}

template<class Cell>
void L2L(Cell &p, Cell &c) {
  complex_t Ynm[P*P], YnmTheta[P*P];
  //const Cell &Cj = C.parent();
  vec3 dX = tovec(c.center() - p.center());
  real_t rho, alpha, beta;

  auto attr = c.attr();

  cart2sph(rho, alpha, beta, dX);
  evalMultipole(rho, alpha, beta, Ynm, YnmTheta);
#if MASS
  attr.L /= attr.M[0];
#endif
  for (int j=0; j<P; j++) {
    for (int k=0; k<=j; k++) {
      int jks = j * (j + 1) / 2 + k;
      complex_t L = 0;
      for (int n=j; n<P; n++) {
        for (int m=j+k-n; m<0; m++) {
          int jnkm = (n - j) * (n - j) + n - j + m - k;
          int nms  = n * (n + 1) / 2 - m;
          L += std::conj(p.attr().L[nms]) * Ynm[jnkm] * real_t(ODDEVEN(k));
        }
        for (int m=0; m<=n; m++) {
          if( n-j >= abs(m-k) ) {
            int jnkm = (n - j) * (n - j) + n - j + m - k;
            int nms  = n * (n + 1) / 2 + m;
            L += p.attr().L[nms] * Ynm[jnkm] * real_t(ODDEVEN((m-k)*(m<k)));
          }
        }
      }
      attr.L[jks] += L;
    }
  }
  c.attr() = attr;
}

} // anon namespace
