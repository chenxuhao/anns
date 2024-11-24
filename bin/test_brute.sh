#!/bin/bash

DATASET=t2i-10M
NP=10
NQ=10000
#DATASET=sift
#NP=1
#NQ=10000

DIM=200
BIN=brute_force_cpu
#BIN=brute_force_gpu

BIN_PATH=~/proj/anns/bin
DATA_PATH=~/proj/anns/data
OUTFILE=output.txt

#DATASET=siftsmall

echo "$BIN_PATH/$BIN $DATA_PATH/$DATASET/base.10M.fbin \
      $DATA_PATH/$DATASET/query.10k.fbin \
      $DATA_PATH/$DATASET/gt.10k.ibin \
      $NP $DIM $NQ $OUTFILE"

$BIN_PATH/$BIN $DATA_PATH/$DATASET/base.10M.fbin \
               $DATA_PATH/$DATASET/query.10k.fbin \
               $DATA_PATH/$DATASET/gt.10k.ibin \
               $NP $DIM $NQ $OUTFILE


