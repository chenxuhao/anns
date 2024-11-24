#!/bin/bash

DATASET=siftsmall
DSIZE=0.01
QSIZE=100
#DATASET=sift
#DSIZE=1
#QSIZE=10000

DIM=128
BIN=ivf_flat_cpu
#BIN=ivf_flat_gpu

DATADIR=/mnt/d/Research/Code/anns-main/data
BINDIR=/mnt/d/Research/Code/anns-main/bin

echo "$BINDIR/$BIN $DATADIR/$DATASET/$DATASET\_base.fvecs 
      $DATADIR/$DATASET/$DATASET\_query.fvecs \
      $DATADIR/$DATASET/$DATASET\_groundtruth.ivecs \
      $DSIZE $DIM $QSIZE output.ivecs"

$BINDIR/$BIN $DATADIR/$DATASET/$DATASET\_base.fvecs \
             $DATADIR/$DATASET/$DATASET\_query.fvecs \
             $DATADIR/$DATASET/$DATASET\_groundtruth.ivecs \
             $DSIZE $DIM $QSIZE output.ivecs
