Tapas - Parallel Framework for Tree-based Adaptively PArtitioned Space

[![Build Status](https://travis-ci.org/keisukefukuda/tapas.svg?branch=master)](https://travis-ci.org/keisukefukuda/tapas)

This software is released under the MIT License, see LICENSE.txt

## How to build Tapas examples

### Compiler Requirement

Tapas requires C++11 compiler. Recommended compiler versions are:

|Compiler               | Version |
|:----------------------|:--------|
|GNU C++ compiler       | >= 4.9  |
|LLVM clang++           | >= 3.6  |
|Intel C++ compiler     | >= 2015 |


### The FMM example 

The most basic compilation

    $ # Non-MPI, serial version
    $ cd sample/exafmm-dev-13274dd4ac68/examples
    $ mpicxx -std=c++11 -O2 -lrt tapas_exafmm.cxx -I../include -I../../../cpp/include -DUSE_MPI -DSpherical -DEXPANSION=10 -DFP64 -o parallel_tapas
    
Depending on how your MPI library is built, the mpicxx compiler may use the default compiler, which does not support C++11.
In such a case, you can speicfy the underlying C++ compiler via environment variables. See the manual of your MPI implementation for details.

    $ # For mpich family (mpcih, mvapich, Intel MPI, etc.)
    $ export MPICH_CXX="your new C++ compiler"
    
## Preprocessor symbols

|Name                   | Possible values  | Default value | Description                                               |
|:----------------------|:-----------------|:--------------|:----------------------------------------------------------|
|TAPAS_DEBUG            | unset, 0, or 1   | unset         | Enable verbose debug output. Serious performance slowdown |
|USE_MPI                | unset/any        | unset         | Use MPI version of Tapas (in ExaFMM example)              | 
|TAPAS_REPORT_PREFIX    | filename prefix  | unset         | Prefix of performance report file names                   |
|TAPAS_REPORT_SUFFIX    | part of filename | unset         | Suffix of performance report file names                   |
|TAPAS_DEBUG_COMM_MATRIX| unset/any        | unset         | Print Communication Matrix in MPI_Alltoallv()             |
|TAPAS_DEBUG_HISTOGRAM  | unset/any        | unset         | Print depth histogram of the tree                         |

