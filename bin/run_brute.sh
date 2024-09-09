#!/bin/bash

DATASET=sift
NP=1
NQ=10000

DIM=128
BIN=brute_force_cpu
BIN=brute_force_gpu

BIN_PATH=../bin
DATA_PATH=../data
OUTFILE=output.txt

echo "$BIN_PATH/$BIN $DATA_PATH/$DATASET/$DATASET\_base.fvecs \
      $DATA_PATH/$DATASET/$DATASET\_query.fvecs \
      $DATA_PATH/$DATASET/$DATASET\_groundtruth.ivecs \
      $NP $DIM $NQ $OUTFILE"

$BIN_PATH/$BIN $DATA_PATH/$DATASET/$DATASET\_base.fvecs \
               $DATA_PATH/$DATASET/$DATASET\_query.fvecs \
               $DATA_PATH/$DATASET/$DATASET\_groundtruth.ivecs \
               $NP $DIM $NQ $OUTFILE
