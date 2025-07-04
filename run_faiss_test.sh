PATH=~/datasets

NAME=sift10M
NAME=deep1M
NAME=gist
NAME=sift
METRIC=l2
FORMAT=vecs
NPROBE=32

echo "./bin/faiss_test vecs data/$NAME/$NAME\_base.fvecs data/$NAME/$NAME\_query.fvecs  data/$NAME/$NAME\_groundtruth.ivecs 100 data/$NAME/$NAME\_ivf_flat_l2.index l2 32"
./bin/faiss_test vecs data/$NAME/$NAME\_base.fvecs data/$NAME/$NAME\_query.fvecs  data/$NAME/$NAME\_groundtruth.ivecs 100 data/$NAME/$NAME\_ivf_flat_l2.index l2 32