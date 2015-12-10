#!/bin/bash
set -u
set -e


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

function get-script-dir() {
    pushd `dirname $0` >/dev/null
    DIR=`pwd`
    popd >/dev/null
    echo $DIR
}

echo --------------------------------------------------------------------
#------------------------------------------------------------------------
# setup
#------------------------------------------------------------------------
TMP_DIR=/tmp/tapas-build
SRC_ROOT=`get-script-dir`

STATUS=0
MAX_ERR="5e-2"

mkdir -p $TMP_DIR
cd $TMP_DIR

echo test.sh
echo Started at $(date)
echo "hostname=" $(hostname)
echo "compiler = ${COMPILER}"
echo "scale = ${SCALE}"

echo date
date

echo pwd
pwd

echo "SRC_ROOT=$SRC_ROOT"
echo "TMP_DIR=$TMP_DIR"

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
    MPICC="env MPICH_CXX=${CXX} MPICH_CC=${CXX} mpicc"
    MPICXX="env MPICH_CXX=${CXX} MPICH_CC=${CXX} mpicxx"
    echo Looks like Mpich.
fi

echo MPICXX=${MPICXX}
echo MPICC=${MPICC}

echo mpicxx -show
mpicxx -show

$CXX --version

echo --------------------------------------------------------------------
echo C++ Unit Tests
echo --------------------------------------------------------------------

SRC_DIR=$SRC_ROOT/cpp/tests

echoCyan make MPICC="${MPICC}" MPICXX="${MPICXX}" VERBOSE=1 MODE=release -C $SRC_DIR clean all
make MPICC="${MPICC}" MPICXX="${MPICXX}" VERBOSE=1 MODE=release -C $SRC_DIR clean all

for t in $SRC_DIR/test_*; do
    if [[ -x $t ]]; then
        echoCyan $t 
        $t 
    fi
done

echo --------------------------------------------------------------------
echo Barnes Hut
echo --------------------------------------------------------------------

MAX_ERR=1e-2

if echo $SCALE | grep -Ei "^t(iny)?" >/dev/null ; then
    NP=(1)
    NB=(100)
elif echo $SCALE | grep -Ei "^s(mall)?" >/dev/null ; then
    NP=(1 4)
    NB=(1000)
elif echo $SCALE | grep -Ei "^m(edium)?" >/dev/null ; then
    NP=(1 2 3 4 5 6)
    NB=(1000 2000)
elif echo $SCALE | grep -Ei "^l(arge)?" >/dev/null ; then
    NP=(1 2 4 8 16 32)
    NB=(1000 2000 4000 8000 16000)
else
    echo "Unknown SCALE : '$SCALE'" >&2
    exit 1
fi
    
SRC_DIR=$SRC_ROOT/sample/barnes-hut
BIN=$SRC_DIR/bh_mpi

make VERBOSE=1 MODE=release -C $SRC_DIR clean all

for np in ${NP[@]}; do
    for nb in ${NB[@]}; do
        echoCyan mpirun -n $np $SRC_DIR/bh_mpi -w $nb
        mpirun -n $np $SRC_DIR/bh_mpi -w $nb | tee log.txt

        PERR=$(grep "P ERR" log.txt | grep -oE "[0-9.e+-]+")
        FERR=$(grep "F ERR" log.txt | grep -oE "[0-9.e+-]+")

        if [[ $(python -c "print $PERR > $MAX_ERR") == "True" ]]; then
            echoRed "*** Error check failed. P ERR $PERR > $MAX_ERR"
            STATUS=$(expr $STATUS + 1)
        else
            echoGreen P ERR OK
        fi
        if [[ $(python -c "print $FERR > $MAX_ERR") == "True" ]]; then
            echoRed "*** Error check failed. F ERR $FERR > $MAX_ERR"
            STATUS=$(expr $STATUS + 1)
        else
            echoGreen F ERR OK
        fi
        echo
        echo
    done
done

echo --------------------------------------------------------------------
echo ExaFMM
echo --------------------------------------------------------------------

MAX_ERR=6e-2

if echo $SCALE | grep -Ei "^t(iny)?" >/dev/null ; then
    NP=(1)
    NB=(100)
    DIST=(c)
    NCRIT=(16)
elif echo $SCALE | grep -Ei "^s(mall)?" >/dev/null ; then
    NP=(1 4)
    NB=(1000)
    DIST=(c)
    NCRIT=(16)
elif echo $SCALE | grep -Ei "^m(edium)?" >/dev/null ; then
    NP=(1 2 3 4 5 6)
    NB=(1000 2000)
    DIST=(l s p c)
    NCRIT=(16 64)
elif echo $SCALE | grep -Ei "^l(arge)?" >/dev/null ; then
    NP=(1 2 4 8 16 32)
    NB=(1000 2000 4000 8000 16000)
    DIST=(l s p c)
    NCRIT=(16 64)
else
    echo "Unknown SCALE : '$SCALE'" >&2
    exit 1
fi
    
SRC_DIR=$SRC_ROOT/sample/exafmm-dev-13274dd4ac68/examples
BIN=$SRC_DIR/bh_mpi

make VERBOSE=1 MODE=release -C $SRC_DIR clean tapas

for dist in ${DIST[@]}; do
    for nb in ${NB[@]}; do
        for ncrit in NCRIT ${NCRIT[@]}; do
            for mutual in 0 1; do
                echoCyan $SRC_DIR/serial_tapas -n $nb -c $ncrit -d $dist --mutual $mutual
                $SRC_DIR/serial_tapas -n $nb -c $ncrit -d $dist --mutual $mutual | tee log.txt
                
                PERR=$(grep "Rel. L2 Error" log.txt | grep pot | sed -e "s/^.*://" | grep -oE "[0-9.e+-]+")
                AERR=$(grep "Rel. L2 Error" log.txt | grep acc | sed -e "s/^.*://" | grep -oE "[0-9.e+-]+")

                if [[ $(python -c "print $PERR > $MAX_ERR") == "True" ]]; then
                    echoRed "*** Error check failed. L2 Error (pot) $PERR > $MAX_ERR"
                    STATUS=$(expr $STATUS + 1)
                else
                    echoGreen pot error OK
                fi
                if [[ $(python -c "print $AERR > $MAX_ERR") == "True" ]]; then
                    echoRed "*** Error check failed. L2 Error (acc) $AERR > $MAX_ERR"
                    STATUS=$(expr $STATUS + 1)
                else
                    echoGreen acc error OK
                fi

                echo
                echo
            done
        done
    done
done

#env CC=$CC CXX=$CXX python test.py

if [[ $STATUS -eq 0 ]]; then
    echo OK.
else
    exit $STATUS
fi
