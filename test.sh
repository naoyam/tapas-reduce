#!/bin/bash

set -eu

echo "TRAVIS = ${TRAVIS}"

# In the Travis's configuration, environmental variable "TRAVIS" is set to 1.
if [ "x${TRAVIS}" = "xtrue" ]; then
    COMPILERS=("clang++-3.5" "g++-5" "g++-4.9")
else
    COMPILERS=("clang++")
fi

for comp in ${COMPILERS[@]}; do
    echo "test.sh: compiler = ${comp}"
    export CXX="${comp}"
    export CC=$(echo $CXX | sed -e 's/clang++/clang/' | sed -e 's/g++/gcc/')
    python test.py
done

echo OK.
exit 0
