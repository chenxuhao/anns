#./bin/faiss_ivf_cpu bin ~/datasets/t2i/base.1M.fbin ~/datasets/t2i/query.public.100K.fbin ~/datasets/t2i/gt100.1M.ibin 100 ~/datasets/t2i/t2i1M_ivf_flat_l2.index l2 64

#./bin/faiss_ivf_cpu bin ~/datasets/t2i/base.1M.fbin ~/datasets/t2i/query.public.100K.fbin ~/datasets/t2i/gt100.1M.ibin 100 ~/datasets/t2i/t2i1M_ivf_flat_ip.index ip 64

#./bin/faiss_ivf_cpu vecs ~/datasets/siftsmall/siftsmall_base.fvecs ~/datasets/siftsmall/siftsmall_query.fvecs ~/datasets/siftsmall/siftsmall_groundtruth.ivecs 100 ~/datasets/siftsmall/siftsmall_ivf_flat.index l2 16

PATH=~/datasets

DATASET=gist
DATASET=sift10m
DATASET=deep1M
DATASET=sift
METRIC=l2
FORMAT=vecs
NPROBE=32

echo "./bin/faiss_ivf_cpu $FORMAT $PATH/$DATASET/$DATASET\_base.fvecs $PATH/$DATASET/$DATASET\_query.fvecs $PATH/$DATASET/$DATASET\_groundtruth.ivecs 100  $PATH/$DATASET/$DATASET\_ivf_flat.index $METRIC $NPROBE"

./bin/faiss_ivf_cpu $FORMAT $PATH/$DATASET/$DATASET\_base.fvecs $PATH/$DATASET/$DATASET\_query.fvecs $PATH/$DATASET/$DATASET\_groundtruth.ivecs 100  $PATH/$DATASET/$DATASET\_ivf_flat.index $METRIC $NPROBE

DATASET=t2i
SIZE=1M
FORMAT=bin
METRIC=ip

#./bin/faiss_ivf_cpu $FORMAT $PATH/$DATASET/base.$SIZE.fbin $PATH/$DATASET/query.public.100K.fbin $PATH/$DATASET/gt100.$SIZE.ibin 100 $PATH/$DATASET/$DATASET$SIZE\_ivf_flat.index $METRIC
