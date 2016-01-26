#!/bin/sh

set -ue

DIR=`dirname $0`

source $DIR/setup.sh

echo $* >&2
$*

