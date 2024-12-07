#!/bin/bash

<<<<<<< HEAD
DATASET=siftsmall
NP=0.01
NQ=100
=======
DATASET=t2i-10M
NP=10
NQ=10000
>>>>>>> a68acf06865f19420bb2b8f90ca3bf80c7212262
#DATASET=sift
#NP=1
#NQ=10000

<<<<<<< HEAD
DIM=128
=======
DIM=200
>>>>>>> a68acf06865f19420bb2b8f90ca3bf80c7212262
BIN=brute_force_cpu
#BIN=brute_force_gpu

BIN_PATH=~/proj/anns/bin
DATA_PATH=~/proj/anns/data
OUTFILE=output.txt

<<<<<<< HEAD
DATASET=siftsmall

echo "$BIN_PATH/$BIN $DATA_PATH/$DATASET/$DATASET\_base.fvecs \
      $DATA_PATH/$DATASET/$DATASET\_query.fvecs \
      $DATA_PATH/$DATASET/$DATASET\_groundtruth.ivecs \
      $NP $DIM $NQ $OUTFILE"

$BIN_PATH/$BIN $DATA_PATH/$DATASET/$DATASET\_base.fvecs \
               $DATA_PATH/$DATASET/$DATASET\_query.fvecs \
               $DATA_PATH/$DATASET/$DATASET\_groundtruth.ivecs \
=======
#DATASET=siftsmall

echo "$BIN_PATH/$BIN $DATA_PATH/$DATASET/base.10M.fbin \
      $DATA_PATH/$DATASET/query.10k.fbin \
      $DATA_PATH/$DATASET/gt.10k.ibin \
      $NP $DIM $NQ $OUTFILE"

$BIN_PATH/$BIN $DATA_PATH/$DATASET/base.10M.fbin \
               $DATA_PATH/$DATASET/query.10k.fbin \
               $DATA_PATH/$DATASET/gt.10k.ibin \
>>>>>>> a68acf06865f19420bb2b8f90ca3bf80c7212262
               $NP $DIM $NQ $OUTFILE


