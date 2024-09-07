#! /bin/bash
DATADIR=../data
BINDIR=../bin

DIM=128
K=32
DATASET=siftsmall
DSIZE=0.01 # million

#DATASET=sift
#DSIZE=1

#DATASET=gist
#DSIZE=1
#DIM=960

BIN="build_graph_cpu"
#BIN="build_graph_gpu"
#../bin/build_knn_graph 32 0.01 128 ../data/siftsmall/siftsmall_base.fvecs ../data/siftsmall/graph-d32

echo "$BINDIR/$BIN $K $DSIZE $DIM $DATADIR/$DATASET/$DATASET\_base.fvecs $DATADIR/$DATASET/graph-nnd-d$K"
$BINDIR/$BIN $K $DSIZE $DIM $DATADIR/$DATASET/$DATASET\_base.fvecs $DATADIR/$DATASET/graph-nnd-d$K

