#!/bin/bash

DATA=siftsmall
GRAPH=graph
NQ=100
NP=0.01

DATA=sift
GRAPH=graph
NQ=10000
NP=1

BIN_PATH=~/proj/anns/bin
DATA_PATH=~/proj/anns/data/$DATA
BIN=parlayann_cpu
#BIN=graph_search_gpu
K=32
DIM=128

echo "$BIN_PATH/$BIN $DATA_PATH/$DATA\_base.fvecs \
  $DATA_PATH/$DATA\_query.fvecs \
  $DATA_PATH/$DATA\_groundtruth.ivecs \
  $NP $DIM $NQ output.txt $DATA_PATH/$GRAPH-d$K"

$BIN_PATH/$BIN $DATA_PATH/$DATA\_base.fvecs \
  $DATA_PATH/$DATA\_query.fvecs \
  $DATA_PATH/$DATA\_groundtruth.ivecs \
  $NP $DIM $NQ output.txt $DATA_PATH/$GRAPH-d$K

