#ifndef dataset_h
#define dataset_h
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "types.h"

// for debug
#include <iomanip>
#include <unistd.h>
#include <sys/syscall.h> // for gettid()
#include <sys/types.h>   // for gettid()
namespace _local {
class Stderr {
  std::ostream *fs_;

 public:
  Stderr(const char *label) : fs_(nullptr) {
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
}

class Dataset {                                                 // Contains all the different datasets
private:
    long filePosition;                                            // Position of file stream
    const unsigned master_seed_;

private:
  //! Split range and return partial range
  void splitRange(int & begin, int & end, int iSplit, int numSplit) {
    assert(end > begin);                                        // Check that size > 0
    int size = end - begin;                                     // Size of range
    int increment = size / numSplit;                            // Increment of splitting
    int remainder = size % numSplit;                            // Remainder of splitting
    begin += iSplit * increment + std::min(iSplit,remainder);   // Increment the begin counter
    end = begin + increment;                                    // Increment the end counter
    if (remainder > iSplit) end++;                              // Adjust the end counter for remainder

    std::cout << "splitRange [" << iSplit << "] "
              << "size = " << size << "\t"
              << "inc = " << increment << "\t"
              << "rem = " << remainder << "\t"
              << "beg = " << begin << "\t"
              << "end = " << end << std::endl;
  }

    /**
     * @brief Generates uniform distribution on [-1,1]^3 lattice
     * @note The first argument numBodies will be overwritten because 
     *       number of lattices does not always match the given numBodies.
     */
    Bodies lattice(int &numBodies, int proc_rank, int proc_size) {
        long nx = std::lround(std::pow(numBodies, 1./3));
        long ny = nx;
        long nz = nx;

        // Overwrite numBodies because the actual number of bodies.size() is not numBodies
        // in many cases.
        numBodies = nx * ny * nz;

        long rem = numBodies % proc_size;
        long nb_local = numBodies / proc_size + (proc_rank + 1 == proc_size ? rem : 0); // local number of bodies
        long beg = (numBodies / proc_size) * proc_rank;
        long end = beg + nb_local;
        Bodies bodies(nb_local);
        size_t gi = 0, li = 0;
        assert(proc_size == 1 || numBodies > nb_local);

        for (int ix = 0; ix < nx; ix++) {
          for (int iy = 0; iy < ny; iy++) {
            for (int iz = 0; iz < nz; iz++) {
              if (beg <= gi  && gi < end) {
                real_t x = (ix / real_t(nx-1)) * 2 - 1;         //    x coordinate
                real_t y = (iy / real_t(ny-1)) * 2 - 1;         //    y coordinate
                real_t z = (iz / real_t(nz-1)) * 2 - 1;         //    z coordinate
                Body &B = bodies[li++];                                       
                B.X[0] = x;
                B.X[1] = y;
                B.X[2] = z;
              }
              gi++;
            }
          }
        }
        assert(li == nb_local);
        assert(gi == numBodies);
        assert(nb_local == bodies.size());
        return bodies;                                              // Return bodies
    }
  
    /**
     * @brief Random distribution in [-1,1]^3 cube
     * Generates cube distribution using the seed given to the constructor.
     * It always generates an identical distribution from a same seed, no matter 
     * how many processes are used.
     */
    Bodies cube(int numBodies, int proc_rank, int proc_size) {
        // Calculate number of bodies which this process generates.
        long rem = numBodies % proc_size;
        long nb_local = numBodies / proc_size + (proc_rank + 1 == proc_size ? rem : 0); // num bodies local
        long beg = (numBodies / proc_size) * proc_rank;
        long end = beg + nb_local;
        Bodies bodies;
        bodies.reserve(nb_local);

        srand48(master_seed_);                                      //  Set seed for random number generator

        for (long i = 0; i < numBodies; i++) {
            Body B;
            for (int d=0; d<3; d++) {                               //   Loop over dimension
                B.X[d] = drand48() * 2 * M_PI - M_PI;               //    Initialize coordinates
            }                                                       //   End loop over dimension
            if (beg <= i && i < end) {
                bodies.push_back(B);                                // Use this particle if it belongs to this process.
            }
        }
        return bodies;
    }

    //! Random distribution on r = 1 sphere
    Bodies sphere(int numBodies, int proc_rank, int proc_size) {
        // Calculate number of bodies which this process generates.
        long rem = numBodies % proc_size;
        long nb_local = numBodies / proc_size + (proc_rank + 1 == proc_size ? rem : 0); // num bodies local
        long beg = (numBodies / proc_size) * proc_rank;
        long end = beg + nb_local;
        Bodies bodies;
        bodies.reserve(nb_local);

        srand48(master_seed_);                                      //  Set seed for random number generator
        
        for (long i = 0; i < numBodies; i++) {
            Body B;
            for (int d=0; d<3; d++) {                               //   Loop over dimension
                B.X[d] = drand48() * 2 - 1;                         //    Initialize coordinates
            }                                                       //   End loop over dimension
            if (beg <= i && i < end) {
                real_t r = std::sqrt(norm(B.X));                        //   Distance from center
                for (int d=0; d<3; d++) {                               //   Loop over dimension
                    B.X[d] /= r * 1.1;                                  //    Normalize coordinates
                }                                                       //   End loop over dimension
                bodies.push_back(B);                                // Use this particle if it belongs to this process.
            }
        }
        assert(bodies.size() == nb_local);
        return bodies;
    }

    //! Plummer distribution in a r = M_PI/2 sphere
    Bodies plummer(int numBodies, int proc_rank, int proc_size) {
        assert(proc_rank < proc_size);
        // Calculate number of bodies which this process generates.
        long rem = numBodies % proc_size;
        long nb_local = numBodies / proc_size + (proc_rank + 1 == proc_size ? rem : 0); // num bodies local
        long beg = (numBodies / proc_size) * proc_rank;
        long end = beg + nb_local;
        Bodies bodies(nb_local);
        //bodies.reserve(nb_local);
        
        srand48(master_seed_);                                      //  Set seed for random number generator
        
        for (long i = 0; i < numBodies; ) {
            real_t X1 = drand48();                                  //   First random number
            real_t X2 = drand48();                                  //   Second random number
            real_t X3 = drand48();                                  //   Third random number
            real_t R = 1.0 / sqrt( (pow(X1, -2.0 / 3.0) - 1.0) );   //   Radius

            // Use this point only if R < 100
            if (R >= 100.0) continue;
            
            if (beg <= i && i < end) {  // Use this particle if it belongs this process
              real_t Z = (1.0 - 2.0 * X2) * R;                      //    z component
              real_t X = sqrt(R * R - Z * Z) * cos(2.0 * M_PI * X3);//    x component
              real_t Y = sqrt(R * R - Z * Z) * sin(2.0 * M_PI * X3);//    y component
              real_t scale = 3.0 * M_PI / 16.0;                     //    Scaling factor
              X *= scale; Y *= scale; Z *= scale;                   //    Scale coordinates
              Body &B=bodies[i-beg];
              B.X[0] = X;                                          //    Assign x coordinate to body
              B.X[1] = Y;                                          //    Assign y coordinate to body
              B.X[2] = Z;                                          //    Assign z coordinate to body
            }
            i++;
        }
        assert(bodies.size() == nb_local);

        return bodies;
    }

public:
  //! Constructor
  Dataset(unsigned seed=0) : filePosition(0), master_seed_(seed) {}                                // Initialize variables

  /**
   * @brief Initialize source values
   * @param numBodies Total number of bodies (over ALL processes)
   * @param bodies Local bodies
   */
  void initSource(Bodies & bodies, int numBodies, int proc_rank, int proc_size) {
    srand48(master_seed_);                                      //  Set seed for random number generator
    
    long rem = numBodies % proc_size;
    long nb_local = numBodies / proc_size + (proc_rank + 1 == proc_size ? rem : 0); // num bodies local
    long beg = (numBodies / proc_size) * proc_rank;
    long end = beg + nb_local;

    assert(nb_local == bodies.size());
    
#if MASS
    for (auto &b : bodies) {
      b.SRC = 1. / numBodies;                                 // Set source values (maybe mass)
    }
#else
    
#if 1
    int idx = 0;
    for (int i = 0; i < numBodies; i++) {
      real_t src = drand48();
      
      if (beg <= i && i < end) {
        Body &b = bodies[idx++];
        b.SRC = src - .5;
      }
    }
#else
    real_t average = 0;
    for (auto &b : bodies) {
      b.SRC = drand48() - .5;
      average += b.SRC;
    }

    average /= bodies.size();
    for (auto &b : bodies) {
      b.SRC -= average;
    }
#endif
#endif // if MASS
  }
    
    //! Initialize target values
    void initTarget(Bodies & bodies) {
        for (auto b = bodies.begin(); b != bodies.end(); b++) {
            b->TRG = 0;                                              //  Clear target values
            b->IBODY = b - bodies.begin();                           //  Initial body numbering
            b->WEIGHT = 1;                                           //  Initial weight
        }                                                            // End loop over bodies
    }

    /**
     * @brief  Initialize dsitribution, source & target value of bodies
     * @note The argument numBodies might be overwritten by the actual size of bodies
     * because bodies.size() is sometimes less than the requested value (numBodies) especially in lattice
     * distribution due to the restriction of body placement.
     */
    Bodies initBodies(int &numBodies, const char * distribution,
                      int mpirank=0, int mpisize=1) {
        Bodies bodies;                                              // Initialize bodies
        switch (distribution[0]) {                                  // Switch between data distribution type
            case 'l':                                                   // Case for lattice
                bodies = lattice(numBodies, mpirank, mpisize);          //  Uniform distribution on [-1,1]^3 lattice
                break;                                                  // End case for lattice
            case 'c':                                                   // Case for cube
                bodies = cube(numBodies, mpirank, mpisize);             //  Random distribution in [-1,1]^3 cube
                break;                                                  // End case for cube
            case 's':                                                   // Case for sphere
                bodies = sphere(numBodies,mpirank,mpisize);             //  Random distribution on surface of r = 1 sphere
                break;                                                  // End case for sphere
            case 'p':                                                   // Case plummer
                bodies = plummer(numBodies,mpirank,mpisize);            //  Plummer distribution in a r = M_PI/2 sphere
                break;                                                  // End case for plummer
            default:                                                    // If none of the above
                fprintf(stderr, "Unknown data distribution %s\n", distribution);// Print error message
        }                                                           // End switch between data distribution type
        initSource(bodies, numBodies, mpirank, mpisize);            // Initialize source values
        initTarget(bodies);                                         // Initialize target values
        return bodies;                                              // Return bodies
    }

  //! Read source values from file
  void readSources(Bodies & bodies, int mpirank) {
    std::stringstream name;                                     // File name
    name << "source" << std::setfill('0') << std::setw(4)       // Set format
         << mpirank << ".dat";                                  // Create file name
    std::ifstream file(name.str().c_str(),std::ios::in);        // Open file
    file.seekg(filePosition);                                   // Set position in file
    for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {       // Loop over bodies
      file >> B->X[0];                                          //  Read data for x coordinates
      file >> B->X[1];                                          //  Read data for y coordinates
      file >> B->X[2];                                          //  Read data for z coordinates
      file >> B->SRC;                                           //  Read data for charge
    }                                                           // End loop over bodies
    filePosition = file.tellg();                                // Get position in file
    file.close();                                               // Close file
  }

  //! Write source values to file
  void writeSources(Bodies & bodies, int mpirank) {
    std::stringstream name;                                     // File name
    name << "source" << std::setfill('0') << std::setw(4)       // Set format
         << mpirank << ".dat";                                  // Create file name
    std::ofstream file(name.str().c_str(),std::ios::out);       // Open file
    for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {       // Loop over bodies
      file << B->X[0] << std::endl;                             //  Write data for x coordinates
      file << B->X[1] << std::endl;                             //  Write data for y coordinates
      file << B->X[2] << std::endl;                             //  Write data for z coordinates
      file << B->SRC  << std::endl;                             //  Write data for charge
    }                                                           // End loop over bodies
    file.close();                                               // Close file
  }

  //! Read target values from file
  void readTargets(Bodies & bodies, int mpirank) {
    std::stringstream name;                                     // File name
    name << "target" << std::setfill('0') << std::setw(4)       // Set format
         << mpirank << ".dat";                                  // Create file name
    std::ifstream file(name.str().c_str(),std::ios::in);        // Open file
    file.seekg(filePosition);                                   // Set position in file
    for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {       // Loop over bodies
      file >> B->TRG[0];                                        //  Read data for potential
      file >> B->TRG[1];                                        //  Read data for x acceleration
      file >> B->TRG[2];                                        //  Read data for y acceleration
      file >> B->TRG[3];                                        //  Read data for z acceleration
    }                                                           // End loop over bodies
    filePosition = file.tellg();                                // Get position in file
    file.close();                                               // Close file
  }

  //! Write target values to file
  void writeTargets(Bodies & bodies, int mpirank) {
    std::stringstream name;                                     // File name
    name << "target" << std::setfill('0') << std::setw(4)       // Set format
         << mpirank << ".dat";                                  // Create file name
    std::ofstream file(name.str().c_str(),std::ios::out);       // Open file
    for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {       // Loop over bodies
      file << B->TRG[0] << std::endl;                           //  Write data for potential
      file << B->TRG[1] << std::endl;                           //  Write data for x acceleration
      file << B->TRG[2] << std::endl;                           //  Write data for y acceleration
      file << B->TRG[3] << std::endl;                           //  Write data for z acceleration
    }                                                           // End loop over bodies
    file.close();                                               // Close file
  }

  //! Downsize target bodies by even sampling
  void sampleBodies(Bodies & bodies, int numTargets) {
    if (numTargets < int(bodies.size())) {                      // If target size is smaller than current
      int stride = bodies.size() / numTargets;                  //  Stride of sampling
      for (int i=0; i<numTargets; i++) {                        //  Loop over target samples
        bodies[i] = bodies[i*stride];                           //   Sample targets
      }                                                         //  End loop over target samples
      bodies.resize(numTargets);                                //  Resize bodies to target size
    }                                                           // End if for target size
  }

  //! Get bodies with positive charges
  Bodies getPositive(Bodies & bodies) {
    Bodies buffer = bodies;                                     // Copy bodies to buffer
    B_iter B2 = buffer.begin();                                 // Initialize iterator of buffer
    for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {       // Loop over bodies
      if (B->SRC >= 0) {                                        //  If source is positive
        *B2 = *B;                                               //   Copy data to buffer
        B2++;                                                   //   Increment iterator
      }                                                         //  End if for positive source
    }                                                           // End loop over bodies
    buffer.resize(B2-buffer.begin());                           // Resize buffer
    return buffer;                                              // Return buffer
  }


  //! Get bodies with negative charges
  Bodies getNegative(Bodies & bodies) {
    Bodies buffer = bodies;                                     // Copy bodies to buffer
    B_iter B2 = buffer.begin();                                 // Initialize iterator of buffer
    for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {       // Loop over bodies
      if (B->SRC < 0) {                                         //  If source is negative
        *B2 = *B;                                               //   Copy data to buffer
        B2++;                                                   //   Increment iterator
      }                                                         //  End if for negative source
    }                                                           // End loop over bodies
    buffer.resize(B2-buffer.begin());                           // Resize buffer
    return buffer;                                              // Return buffer
  }

  template<class T>
  static void Dump(const T &bodies, std::ostream & strm) {
    for(auto b : bodies) {
      strm << std::scientific << std::showpos << b.X[0] << " "
           << std::scientific << std::showpos << b.X[1] << " "
           << std::scientific << std::showpos << b.X[2] << " "
           << std::endl;
    }
  }

  template<class T>
  static void DumpToFile(const T &bodies, const std::string &fname, bool append=false) {
    std::ios_base::openmode mode = std::ios_base::out;
    if (append) mode |= std::ios_base::app;
    std::ofstream ofs;
    ofs.open(fname.c_str(), mode);
        
    assert(ofs.good());
    Dump(bodies, ofs);
    ofs.close();
  }
};
#endif
