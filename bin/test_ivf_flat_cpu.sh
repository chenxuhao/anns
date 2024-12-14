#!/bin/bash

DATASET=t2i-10M
DSIZE=10
QSIZE=10000

DIM=200
BIN=ivf_flat_cpu
#BIN=ivf_flat_gpu
DATADIR=/mnt/d/Research/Code/anns-main/data
BINDIR=/mnt/d/Research/Code/anns-main/bin

echo "$BINDIR/$BIN $DATADIR/$DATASET/base.10M.fbin \ 
      $DATADIR/$DATASET/query.10k.fbin \
      $DATADIR/$DATASET/gt.10k.ibin \
      $DSIZE $DIM $QSIZE output.ivecs"

$BINDIR/$BIN $DATADIR/$DATASET/base.10M.fbin \
             $DATADIR/$DATASET/query.10k.fbin \
             $DATADIR/$DATASET/gt.10k.ibin \
             $DSIZE $DIM $QSIZE output.ivecs
