#!/bin/sh

set -eu

if [ "xtravis" = "x${1:-}" ]; then
    COMPILERS=("clang++-3.5" "g++-5" "g++-4.9")
else
    COMPILERS=("clang++")
fi

for comp in ${COMPILERS[@]}; do
    echo "test.sh: compiler = ${comp}"
    env CXX="${comp}" python test.py
done

echo OK.
