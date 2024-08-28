#!/bin/bash

BIN_PATH=../bin
DATA_PATH=../data/siftsmall

echo "$BIN_PATH/quantized_search_cpu $DATA_PATH/siftsmall_base.fvecs \
  $DATA_PATH/siftsmall_query.fvecs \
  $DATA_PATH/siftsmall_groundtruth.ivecs \
  0.01 128 100 output.txt"

$BIN_PATH/quantized_search_cpu $DATA_PATH/siftsmall_base.fvecs \
  $DATA_PATH/siftsmall_query.fvecs \
  $DATA_PATH/siftsmall_groundtruth.ivecs \
  0.01 128 100 output.txt

