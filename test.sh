#!/bin/bash
set -u
set -e

unset TMPFILE

atexit() {
    [[ -n "${TMPFILE-}" ]] && rm -f "$TMPFILE"
}

# To trap an error caused by "set -e"
onerror()
{
    status=$?
    script=$0
    line=$1
    shift

    args=
    for i in "$@"; do
        args+="\"$i\" "
    done

    echo ""
    echo "------------------------------------------------------------"
    echo "Error occured on $script [Line $line]: Status $status"
    echo ""
    echo "PID: $$"
    echo "User: $USER"
    echo "Current directory: $PWD"
    echo "Command line: $script $args"
    echo "------------------------------------------------------------"
    echo ""
}

trap atexit EXIT
trap 'trap - EXIT; atexit; exit -1' INT PIPE TERM
trap 'onerror $LINENO "$@"' ERR


TMPFILE=$(mktemp "/tmp/${0##*/}.tmp.XXXXXXX")
echo TMPFILE=${TMPFILE}


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

function get_script_dir() {
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
SRC_ROOT=`get_script_dir`

if [[ -z "${SCALE-}" ]]; then
    SCALE=s
fi

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
    MPICC="env OMPI_CC=${CC} mpicc"
    MPICXX="env OMPI_CXX=${CXX} mpicxx"
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

echo --------------------------------------------------------------------
echo C++ Unit Tests
echo --------------------------------------------------------------------

SRC_DIR=$SRC_ROOT/cpp/tests

echoCyan make MPICC=\"${MPICC}\" MPICXX=\"${MPICXX}\" VERBOSE=1 MODE=release -C $SRC_DIR clean all
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

echoCyan make CXX=\"${CXX}\" CC=\"${CC}\" MPICC=\"${MPICC}\" MPICXX=\"${MPICXX}\" VERBOSE=1 MODE=release -C $SRC_DIR clean all
make CXX=${CXX} CC=${CC} MPICC="${MPICC}" MPICXX="${MPICXX}" VERBOSE=1 MODE=release -C $SRC_DIR clean all

for np in ${NP[@]}; do
    for nb in ${NB[@]}; do
        echoCyan mpiexec -n $np $SRC_DIR/bh_mpi -w $nb
        mpiexec -n $np $SRC_DIR/bh_mpi -w $nb | tee $TMPFILE

        PERR=$(grep "P ERR" $TMPFILE | grep -oE "[0-9.e+-]+")
        FERR=$(grep "F ERR" $TMPFILE | grep -oE "[0-9.e+-]+")

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

MAX_ERR=5e-3

if echo $SCALE | grep -Ei "^t(iny)?" >/dev/null ; then
    NP=(1)
    NB=(100)
    DIST=(c)
    NCRIT=(16)
elif echo $SCALE | grep -Ei "^s(mall)?" >/dev/null ; then
    NP=(1 2)
    NB=(1000)
    DIST=(c)
    NCRIT=(16)
elif echo $SCALE | grep -Ei "^m(edium)?" >/dev/null ; then
    NP=(1 2 3 4 5 6)
    NB=(10000 20000)
    DIST=(l s p c)
    NCRIT=(16 64)
elif echo $SCALE | grep -Ei "^l(arge)?" >/dev/null ; then
    NP=(1 2 4 8 16 32)
    NB=(10000 20000 40000 80000 160000)
    DIST=(l s p c)
    NCRIT=(16 64)
else
    echo "Unknown SCALE : '$SCALE'" >&2
    exit 1
fi
    
SRC_DIR=$SRC_ROOT/sample/exafmm-dev-13274dd4ac68/examples

make VERBOSE=1 MODE=release -C $SRC_DIR clean tapas

function accuracyCheck() {
    local fname=$1
    PERR=$(grep "Rel. L2 Error" $fname | grep pot | sed -e "s/^.*://" | grep -oE "[0-9.e+-]+")
    AERR=$(grep "Rel. L2 Error" $fname | grep acc | sed -e "s/^.*://" | grep -oE "[0-9.e+-]+")

    if [[ $(python -c "print $PERR > $MAX_ERR") == "True" ]]; then
        echoRed "*** Error check failed. L2 Error (pot) $PERR > $MAX_ERR"
        STATUS=$(expr $STATUS + 1)
    else
        echoGreen pot check OK
    fi
    if [[ $(python -c "print $AERR > $MAX_ERR") == "True" ]]; then
        echoRed "*** Error check failed. L2 Error (acc) $AERR > $MAX_ERR"
        STATUS=$(expr $STATUS + 1)
    else
        echoGreen acc check OK
    fi
}

for dist in ${DIST[@]}; do
    for nb in ${NB[@]}; do
        for ncrit in ${NCRIT[@]}; do
            for mutual in 0 1; do
                rm -f $TMPFILE; sleep 1s

                # We no longer check serial_tapas
                echoCyan $SRC_DIR/serial_tapas -n $nb -c $ncrit -d $dist --mutual $mutual
                $SRC_DIR/serial_tapas -n $nb -c $ncrit -d $dist --mutual $mutual > $TMPFILE
                cat $TMPFILE

                accuracyCheck $TMPFILE
                
                echo
                echo

                for np in ${NP[@]}; do
                    rm -f $TMPFILE; sleep 1s
                    echoCyan mpiexec -n $np $SRC_DIR/parallel_tapas -n $nb -c $ncrit -d $dist --mutual $mutual
                    mpiexec -n $np $SRC_DIR/parallel_tapas -n $nb -c $ncrit -d $dist --mutual $mutual > $TMPFILE
                    cat $TMPFILE

                    accuracyCheck $TMPFILE
                done

                echo
                echo
            done
        done
    done
done

if [[ $STATUS -eq 0 ]]; then
    echo OK.
fi
exit $STATUS
#env CC=$CC CXX=$CXX python test.py

# Check the GPU version if nvcc is available
if which nvcc >/dev/null 2>&1; then
    NVCC_OPT="-O3 -Xcicc -Xptas --compiler-options -Wall --compiler-options -Wextra --compiler-options -Wno-unused-parameter \
            -Xcompiler -rdynamic -lineinfo --device-debug -x cu -arch sm_35 -ccbin=g++"
    
    which g++
    BIN=parallel_tapas_cuda
    SRC_DIR=$SRC_ROOT/sample/exafmm-dev-13274dd4ac68/examples
    compile=$($MPICXX -show -cxx=nvcc -DTAPAS_DEBUG=0 -DUSE_MPI -g $NVCC_OPT \
                      -DASSERT -DTAPAS_USE_VECTORMAP -DFP64 -DSpherical -DEXPANSION=6 -DTAPAS_LOG_LEVEL=0 \
                      -I$SRC_DIR/../include -I$SRC_DIR -I$SRC_DIR/../../../cpp/include \
                      -std=c++11 $SRC_DIR/../kernels/LaplaceP2PCPU.cxx $SRC_DIR/../kernels/LaplaceSphericalCPU.cxx \
                      $SRC_DIR/tapas_exafmm.cxx -o $BIN)

    # nvcc doesn't spport -Wl,... options
    compile=$(echo $compile | sed -e 's/-Wl,[^ ]*//g')
    echo $compile
    $compile
    
    for dist in ${DIST[@]}; do
        for nb in ${NB[@]}; do
            for ncrit in ${NCRIT[@]}; do
                echoCyan mpiexec -n 1 ./$BIN --numBodies 1000
                mpiexec -n 1 ./$BIN --numBodies 10000 > $TMPFILE
                cat $TMPFILE
                accuracyCheck $TMPFILE
            done
        done
    done

fi

