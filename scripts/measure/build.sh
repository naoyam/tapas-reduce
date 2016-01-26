#!/bin/sh
set -ue

DIR=`dirname $0`

source $DIR/setup.sh

if [[ ! -d $WORK_DIR ]]; then
    echo "Please set WORK_DIR"  >&2
    exit 1
fi

/bin/rm -f $WORK_DIR/*

cd $TAPAS/sample/exafmm-dev-13274dd4ac68/examples
{
    # Check if the workking set is clean
    if ! git diff --quiet ; then
        echo "Your working set is not clean." >&2
        exit 1
    fi

    if ! git diff --cached --quiet ; then
        echo "Your working set is not clean." >&2
        exit 1
    fi
    
    make -C .. clean
    rm -f parallel_tapas
    env MODE=release MYTH_DIR=$HOME/.wrenchset/gcc-4.9.3 CXX=icpc make MTHREAD=1 parallel_tapas
} 1>&2

echo $PWD/parallel_tapas

