#include <cassert>
#include <cstdlib>
#include <iostream>
#include <mpi.h>

/*
  memo:
  MPI_ACCUMULATE
  MPI_COMPARE_AND_SWAP
 */

void test_put() {
   int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // It might be better to use MPI_Alloc_allocate()
    // (MPI_Win_allocate() does both of new and MPI_Win_create()
    int *data; // = new int[size];
    MPI_Win win;

    MPI_Alloc_mem(size * sizeof(data[0]), MPI_INFO_NULL, &data);
    MPI_Win_create(data,                    // base
                   rank == 0 ? sizeof(data[0]) * size : 0,  // size (size in bytes)
                   sizeof(data[0]),         // disp_unit
                   MPI_INFO_NULL,           // info
                   MPI_COMM_WORLD,          // comm
                   &win);                   // win

    data[rank] = rank;
    MPI_Win_fence(0, win);
    //MPI_Win_fence(MPI_MODE_NOPRECEDE,win);
    
    // Perform RPUT operation
    if (rank > 0) {
        int r2 = rank * rank;
        MPI_Put(&r2,     // origin_addr
                1,         // origin_count
                MPI_INT,   // origin_datatype
                0,         // target_rank
                rank,      // target_disp
                1,         // target_count
                MPI_INT,   // target_datatype
                win);      // win
    }
    MPI_Win_fence(0, win);
    
    if (rank == 0) {
        int ok = 0;
        int ng = 0;
        // Check result
        for(int i = 0; i < size; i++) {
            if(data[i] == i*i) {
                ok++;
            } else {
                ng++;
            }
        }
        if (ng == 0 && ok == size) {
            std::cout << "test_put OK" << std::endl;
        } else {
            std::cout << "test_put ERROR!!!" << std::endl;
        }
    }

    // Test the MPI_Get interface.
    // Rank i process gets the (i+1)-th element of the array 'data'
    // and checks if the value is correct.
    int idx = (rank + 1) % size;
    int val = -1;
    MPI_Get(&val,          // origin_addr
            1,             // origin_count
            MPI_INT,       // origin_datatype,
            0,             // target_rank
            idx,          // target_disp
            1,             // target_count
            MPI_INT,       // target_datatype
            win);

    MPI_Win_fence(0, win);

    assert(val == idx * idx);
    
    MPI_Win_free(&win);
    MPI_Free_mem(data);
}

int main(int argc, char**argv) {
    MPI_Init(&argc, &argv);

    test_put();
    
    MPI_Finalize();
    return 0;
}

