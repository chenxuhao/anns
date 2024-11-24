#!/bin/bash

DATASET=siftsmall
DSIZE=0.01
QSIZE=100

DIM=128
BIN=ivf_flat_cpu
#BIN=ivf_flat_gpu
DATADIR=~/proj/anns/data
BINDIR=~/proj/anns/bin

echo "$BINDIR/$BIN $DATADIR/$DATASET/$DATASET\_base.fvecs 
      $DATADIR/$DATASET/$DATASET\_query.fvecs \
      $DATADIR/data/$DATASET/$DATASET\_groundtruth.ivecs \
      $DSIZE $DIM $QSIZE output.ivecs"

$BINDIR/$BIN $DATADIR/$DATASET/$DATASET\_base.fvecs \
             $DATADIR/$DATASET/$DATASET\_query.fvecs \
             $DATADIR/$DATASET/$DATASET\_groundtruth.ivecs \
             $DSIZE $DIM $QSIZE output.ivecs
