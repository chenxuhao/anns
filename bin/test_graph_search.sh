#!/bin/bash

BIN_PATH=../bin
DATA_PATH=../data/siftsmall
K=32

echo "$BIN_PATH/graph_search_cpu $DATA_PATH/siftsmall_base.fvecs \
  $DATA_PATH/siftsmall_query.fvecs \
  $DATA_PATH/siftsmall_groundtruth.ivecs \
  0.01 128 100 output.txt $DATA_PATH/graph-nnd-d$K"

$BIN_PATH/graph_search_cpu $DATA_PATH/siftsmall_base.fvecs \
  $DATA_PATH/siftsmall_query.fvecs \
  $DATA_PATH/siftsmall_groundtruth.ivecs \
  0.01 128 100 output.txt $DATA_PATH/graph-nnd-d$K

