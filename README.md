Tapas - Parallel Framework for Tree-based Adaptively PArtitioned Space

[![Build Status](https://travis-ci.org/keisukefukuda/tapas.svg?branch=master)](https://travis-ci.org/keisukefukuda/tapas)

This software is released under the MIT License, see LICENSE.txt

## Preprocessor symbols

|Name                   | Possible values  | Default value | Description                                               |
|:----------------------|:-----------------|:--------------|:----------------------------------------------------------|
|TAPAS_DEBUG            | unset, 0, or 1   | unset         | Enable verbose debug output. Serious performance slowdown |
|USE_MPI                | unset/any        | unset         | Use MPI version of Tapas (in ExaFMM example)              | 
|TAPAS_REPORT_PREFIX    | filename prefix  | unset         | Prefix of performance report file names                   |
|TAPAS_REPORT_SUFFIX    | part of filename | unset         | Suffix of performance report file names                   |
|TAPAS_DEBUG_COMM_MATRIX| unset/any        | unset         | Print Communication Matrix in MPI_Alltoallv()             |

