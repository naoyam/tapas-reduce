#!/bin/sh
set -ue

DIR=`dirname $0`

source $DIR/setup.sh

if [[ ! -d $WORK_DIR ]]; then
    echo "Please set WORK_DIR"  >&2
    exit 1
fi

/bin/rm -f $WORK_DIR/*

cd $EXAFMM
{
    MYTH_DIR=$HOME/.wrenchset/gcc-4.9.3
    make distclean &&:
    export CXX=g++; export CC=gcc
    #export CXX=icpc; export CC=icc
    set -x
    env CXXFLAGS="-I$MYTH_DIR/include" LDFLAGS="-L$MYTH_DIR/lib" \
        MPICXX="mpicxx -cxx=${CXX}" ./configure --enable-mpi --disable-simd --with-mthread
    make -j4 -C examples laplace_spherical
    set +x
} 1>&2

