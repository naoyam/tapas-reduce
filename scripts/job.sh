#!/bin/sh
# Jobscript for TSUBAME2.5 for SC 16 submission

eval `python ~/wrenchset/wrenchset.py use gcc 4.9.3`
GCC_PREFIX=/home/usr1/13D37066/.wrenchset/gcc-4.9.3/

# source set_ompi-1.8.2_i2013.1.046_cuda7.5.sh
source set_mpch-3.1_i2013.1.046.sh

source /usr/apps.sp3/isv/intel/ParallelStudioXE/ClusterEdition/2016-Update2/compilers_and_libraries_2016.2.181/linux/bin/compilervars.sh intel64

echo LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

declare -a NP=(10 20 30 40)
declare -r MYTH_WORKER_NUM=12
declare -r NC=64
declare -r CHECK=0
export MYTH_WORKER_NUM
export NC

set -u

#--------------------------------------------
# MPI
#--------------------------------------------
if [[ -z "${MPI:-}" ]]; then
    "Please speicfy MPI type (openmpi|mpich)"
    exit -1
fi

MPI=$(tr '[A-Z]' '[a-z]' <<< ${MPI:-})

if [[ "$MPI" = "ompi" || "$MPI" = "openmpi" ]]; then
    declare -r MPIEXEC=/usr/apps.sp3/mpi/openmpi/1.8.2/i2013.1.046_cuda7.5/bin/mpiexec
    declare -r MPI=openmpi
    declare -r MAP=""
elif [[ "$MPI" = "mpich" ]]; then
    #declare -r MPIEXEC=/usr/apps.sp3/isv/intel/ParallelStudioXE/ClusterEdition/2016-Update2/compilers_and_libraries_2016.2.181/linux/mpi/intel64/bin/mpirun
    declare -r MPIEXEC=/usr/apps.sp3/mpi/mpich2/3.1/i2013.1.046/bin/mpiexec
    declare -r MPI=mpich
    declare -r MAP=""
fi

function main() {
    TAPAS=$HOME/tapas

    echo Script $(basename $0) started at $(date)
    echo "PWD=${PWD}"
    echo TAPAS=${TAPAS}

    # check the number of running/queued jobs
    declare -r NJOBS=$(t2stat | grep -vE "[-]{10}" | grep -v "Job id" | grep -vE "^$" | wc -l)
    declare -r JOB_LIMIT=15

    echo "${NJOBS} are currently running"

    if [[ "${NJOBS}" -gt ${JOB_LIMIT} ]]; then
        echo "More than ${JOB_LIMIT} jobs are running. good bye."
        exit 1
    fi

    check

    if [[ -d "${PBS_O_WORKDIR:-}" ]] ; then
        echo PBS_O_WORKDIR=${PBS_O_WORKDIR}
        echo Starting the application.
        run 
    else
        echo 'Submitting a job'
        submit
    fi
}

function show_repostory() {
    pushd $TAPAS

    echo "-------------------------------------"
    echo "Tapas repository information"
    echo "-------------------------------------"
    echo
    pwd
    git status -uno
    echo branch=$(git rev-parse --abbrev-ref HEAD)

    popd
}

function check() {
    if [[ -z "$QUEUE" ]]; then
        QUEUE="S"
    else
        echo "QUEUE=${QUEUE}"
    fi

    if [[ -z "$NB" ]]; then
        echo "Error: NB is empty" >&2
        exit 1
    else
        echo "NB=${NB}"
    fi
    
    if [[ -z "$WTIME" ]]; then
        echo "Error: WTIME is empty" >&2
        exit 1
    else
        echo "WTIME=${WTIME}"
    fi
}

function submit() {
    echo "-------------------------------------"
    echo "Submit"
    echo "-------------------------------------"
    echo

    NP_max=$(ruby -e "puts ARGV.map{|i| i.to_i}.max" ${NP[@]})
    echo NP_max=${NP_max}

    if [[ -z "$GROUP" ]]; then
        echo "Warning: GROUP is empty" >&2
        declare -r GROUP_LIST=""
    else
        echo "GROUP=${GROUP}"
        declare -r GROUP_LIST="-W group_list=${GROUP}"
    fi

    if [[ ! -z "${JOBNAME:-}" ]]; then
      JOBNAME="-N ${JOBNAME}"
    else
      JOBNAME=
    fi
    
    set -x
    t2sub ${JOBNAME} -j oe -V -q $QUEUE ${GROUP_LIST} -l select=${NP_max}:mpiprocs=1:ncpus=12:mem=50gb -l walltime=${WTIME} -et 1 $0
    set +x
}

function run() {
    echo "-------------------------------------"
    echo "Run"
    echo "-------------------------------------"
    echo
    echo PBS_NODEFILE=${PBS_NODEFILE}
    
    # declare NP=$(wc -l $PBS_NODEFILE | grep -oE '^[0-9]+')
    echo NP=${NP}

    cd ${PBS_O_WORKDIR}
    
    show_repostory

    if [[ ! -z "${LABEL:-}" ]]; then
        export TAPAS_REPORT_PREFIX="${LABEL}_"
    else
        export TAPAS_REPORT_PREFIX=""
    fi
    

    if [[ ! -z "${EXAFMM:-}" ]]; then
      declare -r BIN=$TAPAS/sample/exafmm-dev-13274dd4ac68/examples/parallel
    else
      declare -r BIN=$TAPAS/sample/exafmm-dev-13274dd4ac68/examples/parallel_tapas
    fi

    if [[ ! -x "$BIN" ]]; then
        "ERROR: $BIN is not executable"
        exit 1
    fi

    ldd $BIN

    if [[ -z "${NSTEP:-}" ]]; then
        NSTEP=10
    fi

    # set -x
    # echo NP=1
    # export TAPAS_REPORT_SUFFIX="_1"
    # ${MPIEXEC} --map-by node:pe=${MYTH_WORKER_NUM}:nooversubscribe \
    #            --hostfile ${PBS_NODEFILE} \
    #            -x TAPAS_REPORT_SUFFIX \
    #            -x MYTH_WORKER_NUM \
    #            -x TAPAS_REPORT_SUFFIX \
    #            -x TAPAS_REPORT_PREFIX \
    #            -x LD_LIBRARY_PATH \
    #            -x PATH \
    #            -np 1 \
    #            ${BIN} --numBodies ${NB} --ncrit ${NC} --threads=${MYTH_WORKER_NUM} --check ${CHECK} ||:
    # set +x

    for i in ${NP[@]}; do
        export TAPAS_REPORT_SUFFIX="_${i}"
        echo NP=$i
        if [[ "$MPI" = "openmpi" ]]; then
            set -x
            ${MPIEXEC} --map-by node:pe=${MYTH_WORKER_NUM}:nooversubscribe \
                       --hostfile ${PBS_NODEFILE} \
                       -x TAPAS_REPORT_SUFFIX \
                       -x MYTH_WORKER_NUM \
                       -x TAPAS_REPORT_SUFFIX \
                       -x TAPAS_REPORT_PREFIX \
                       -x LD_LIBRARY_PATH \
                       -x PATH \
                       -np $i \
                       ${BIN} --numBodies $(expr $i \* ${NB}) --ncrit ${NC} --threads=${MYTH_WORKER_NUM} --check ${CHECK} ||:
            set +x
        else
            # mpich
            set -x
            env ${MPIEXEC} \
                -f ${PBS_NODEFILE} \
                -map-by node\
                -envall \
                -np $i \
                ${BIN} --numBodies $(expr $i \* ${NB}) --ncrit ${NC} --threads=${MYTH_WORKER_NUM} --check ${CHECK} ||:
            set +x
        fi
    done
}

main
