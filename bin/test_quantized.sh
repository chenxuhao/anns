#!/bin/bash

BIN=quantized_search_cpu
BIN_PATH=../bin
DATA_PATH=../data
OUTFILE=output.txt

DATASET=siftsmall
NP=0.01
DIM=128
NQ=100

echo "$BIN_PATH/$BIN $DATA_PATH/$DATASET/$DATASET\_base.fvecs \
      $DATA_PATH/$DATASET/$DATASET\_query.fvecs \
      $DATA_PATH/$DATASET/$DATASET\_groundtruth.ivecs \
      $NP $DIM $NQ $OUTFILE"

$BIN_PATH/$BIN $DATA_PATH/$DATASET/$DATASET\_base.fvecs \
               $DATA_PATH/$DATASET/$DATASET\_query.fvecs \
               $DATA_PATH/$DATASET/$DATASET\_groundtruth.ivecs \
               $NP $DIM $NQ $OUTFILE


