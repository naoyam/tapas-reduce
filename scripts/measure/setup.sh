if [[ ! -d ${TAPAS:-} ]]; then
    export TAPAS=$HOME/tapas
fi

if [[ ! -d ${EXAFMM:-} ]]; then
    export EXAFMM=$HOME/exafmm
fi

eval `python ~/wrenchset/wrenchset.py use gcc 4.9.3` &&:

# MVAPICH
#source set_mvp-2.0rc1_i2013.1.046_cuda7.0.sh &&:
#export CC="mpicc -cc=icc"
#export CXX="mpicxx -cxx=icpc"

# MPICH
#source set_mpch-3.1_i2013.1.046.sh &&:
#export CC="mpicc -cc=icc"
#export CXX="mpicxx -cxx=icpc"

# Open MPI
source set_ompi-1.8.2_i2013.1.046_cuda7.0.sh &&:
export CC=icc
export CXX=icpc
export MPICC=mpicc
export MPICXX=mpicxx

set +ue
source /usr/apps.sp3/isv/intel/2015.2.164/composer_xe_2015.2.164/bin/compilervars.sh intel64 >/dev/null 2>&1 
set -ue

if [[ -z ${WORK_DIR:-} ]]; then
    WORK_DIR=/tmp/13D37066/
fi
    
if [[ ! -d $WORK_DIR ]]; then
    mkdir -p $WORK_DIR
fi
export WORK_DIR

# {
#     echo "--------------------------------------------------"
#     echo setup.sh
#     echo mpicxx=`which mpicxx`
#     echo mpiexec=`which mpiexec`
#     echo "--------------------------------------------------"
# } >&2
