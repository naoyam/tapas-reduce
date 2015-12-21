#!/bin/bash

set -eu

# -- setup TMPFILE
unset TMPFILE
unset TMPDIR

atexit() {
    [[ -n "${TMPFILE-}" ]] && rm -f "$TMPFILE"
    [[ -d "${TMPDIR-}" ]] && rm -rf "$TMPDIR"
}

trap atexit EXIT
trap 'trap - EXIT; atexit; exit -1' INT PIPE TERM

TMPFILE=$(mktemp "/tmp/${0##*/}.tmp.XXXXXXX")
echo TMPFILE=${TMPFILE}

# -- Setup TMPDIR and cd to it
TMPDIR=/tmp/tapas-build

function echoRed() {
    echo -en "\e[0;31m"
    echo "$*"
    echo -en "\e[0;39m"
}

function echoGreen() {
    echo -en "\e[0;32m"
    echo "$*"
    echo -en "\e[0;39m"
}

function echoCyan() {
    echo -en "\e[0;36m"
    echo "$*"
    echo -en "\e[0;39m"
}

# -- Setup SRC_DIR

function get_script_dir() {
    pushd `dirname $0` >/dev/null
    DIR=`pwd`
    popd >/dev/null

    # Find tapas/ directory by moving up
    while [[ $DIR != "/" ]]; do
        if ls $DIR/cpp/include/tapas.h >/dev/null 2>&1; then
            break
        else
            DIR=$(sh -c "cd $DIR/..; pwd")
        fi
    done

    if [[ $DIR == "/" ]]; then
        echo "ERROR: Can't find tapas/ directory"
        exit 1
    fi

    if [[ ! -d $DIR ]]; then
        echo "ERROR: Something is wrong"
        exit 1
    fi
        
    echo $DIR
}

TAPAS_ROOT=$(get_script_dir)

mkdir -p $TMPDIR
cd $TMPDIR

if [[ -z "${SCALE-}" ]]; then
    SCALE=s
fi

echo Started at $(date)
echo "hostname=" $(hostname)
echo "compiler = ${COMPILER}"
echo "scale = ${SCALE}"

echo date
date

echo pwd
pwd

echo "TAPAS_ROOT=$TAPAS_ROOT"
echo "TMPDIR=$TMPDIR"

echo "ls /usr/bin/gcc*"
ls /usr/bin/gcc*
echo "ls /usr/bin/g++*"
ls /usr/bin/g++*

echo PATH=$PATH

export CXX=$COMPILER
export CC=$(echo $CXX | sed -e 's/clang++/clang/' | sed -e 's/g++/gcc/' | sed -e 's/icpc/icc/')

echo CC=$(which ${CC})
echo CXX=$(which ${CXX})
echo MPICXX=$(which mpicxx)

echo Detecting mpicxx implementation

# detect MPI implementation
if mpicxx --showme:version 2>/dev/null | grep "Open MPI"; then
    # Opne MPI
    MPICC="env CC=${CC} CXX=${CXX} mpicc"
    MPICXX="env CC=${CC} CXX=${CXX} mpicxx"
    echo Looks like Open MPI.
else
    # mpich family (mpich and mvapich)
    MPICC="env MPICH_CXX=${CXX} MPICH_CC=${CC} mpicc"
    MPICXX="env MPICH_CXX=${CXX} MPICH_CC=${CC} mpicxx"
    echo Looks like Mpich.
fi

echo MPICXX=${MPICXX}
echo MPICC=${MPICC}

echo mpicxx -show
mpicxx -show

$CXX --version


