#!/bin/bash

DATA=sift
NQ=10000
NP=1
GRAPH=graph

DATA=siftsmall
NQ=100
NP=0.01
GRAPH=graph

K=32
DIM=128

BIN_PATH=~/proj/anns/bin
DATA_PATH=~/proj/anns/data/$DATA
BIN=graph_search_cpu
#BIN=graph_search_gpu

echo "$BIN_PATH/$BIN $DATA_PATH/$DATA\_base.fvecs \
  $DATA_PATH/$DATA\_query.fvecs \
  $DATA_PATH/$DATA\_groundtruth.ivecs \
  $NP $DIM $NQ output.txt $DATA_PATH/$GRAPH-d$K"

$BIN_PATH/$BIN $DATA_PATH/$DATA\_base.fvecs \
  $DATA_PATH/$DATA\_query.fvecs \
  $DATA_PATH/$DATA\_groundtruth.ivecs \
  $NP $DIM $NQ output.txt $DATA_PATH/$GRAPH-d$K

