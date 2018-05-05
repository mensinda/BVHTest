#!/bin/bash

cd $(dirname $0)

BIN_FILE=./build/src/bvhTest

perf record -g $BIN_FILE run $*
perf report -g 'graph,0.5,caller' -i perf.data

rm perf.data
