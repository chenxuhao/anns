#!/bin/bash

DATASET=siftsmall
DSIZE=0.01
QSIZE=100
#DATASET=sift
#DSIZE=1
#QSIZE=10000

DIM=128
BIN=brute_force_cpu

echo "../bin/$BIN ./../data/$DATASET/$DATASET\_base.fvecs ./../data/$DATASET/$DATASET\_query.fvecs ./../data/$DATASET/$DATASET\_groundtruth.ivecs $DSIZE $DIM $QSIZE output.ivecs"
../bin/$BIN ./../data/$DATASET/$DATASET\_base.fvecs ./../data/$DATASET/$DATASET\_query.fvecs ./../data/$DATASET/$DATASET\_groundtruth.ivecs $DSIZE $DIM $QSIZE output.ivecs
