PATH=~/datasets

#./bin/faiss_hnsw bin ~/datasets/t2i/base.1M.fbin ~/datasets/t2i/query.public.100K.fbin ~/datasets/t2i/gt100.1M.ibin 100 ~/datasets/t2i/t2i1M_hnsw_ip_M32.index ip 32 128

#./bin/faiss_hnsw bin ~/datasets/t2i/base.1M.fbin ~/datasets/t2i/query.public.100K.fbin ~/datasets/t2i/gt100.1M.ibin 100 ~/datasets/t2i/t2i1M_hnsw_l2_M32.index l2 32

#./bin/faiss_hnsw bin ~/datasets/t2i/base.10M.fbin ~/datasets/t2i/query.public.100K.fbin ~/datasets/t2i/gt100.10M.ibin 100 ~/datasets/t2i/t2i10M_hnsw_l2_M32.index l2 32

#./bin/faiss_hnsw bin ~/datasets/t2i/base.10M.fbin ~/datasets/t2i/query.public.100K.fbin ~/datasets/t2i/gt100.10M.ibin 100 ~/datasets/t2i/t2i10M_hnsw_ip_M32.index ip 32

#./bin/faiss_hnsw vecs ~/datasets/sift10m/sift10m_base.fvecs ~/datasets/sift10m/sift10m_query.fvecs ~/datasets/sift10m/sift10m_groundtruth.ivecs 100 ~/datasets/sift10m/sift10m_hnsw_l2_M32.index l2 32

#./bin/faiss_hnsw vecs ~/datasets/sift/sift_base.fvecs ~/datasets/sift/sift_query.fvecs ~/datasets/sift/sift_groundtruth.ivecs 100 ~/datasets/sift/sift_hnsw_l2_M32.index l2 32

#./bin/faiss_hnsw u8bin ~/datasets/bigann/base.1B.u8bin.crop_nb_10000000 ~/datasets/bigann/query.public.10K.u8bin ~/datasets/bigann/bigann-10M 100 ~/datasets/bigann/bigann10M_hnsw_l2_M32 l2 32

FORMAT=vecs
DATASET=sift10m
DATASET=gist
DATASET=deep1M
DATASET=deep10M
DATASET=sift

DSIZE=100M
DSIZE=1M

DSIZE=10M
QSIZE=10K

METRIC='l2'
DEGREE=32

echo "./bin/faiss_hnsw $FORMAT $PATH/$DATASET/$DATASET\_base.fvecs $PATH/$DATASET/$DATASET\_query.fvecs $PATH/$DATASET/$DATASET\_groundtruth.ivecs 100 $PATH/$DATASET/$DATASET\_hnsw_$METRIC\_M$DEGREE.index $METRIC $DEGREE"
./bin/faiss_hnsw $FORMAT $PATH/$DATASET/$DATASET\_base.fvecs $PATH/$DATASET/$DATASET\_query.fvecs $PATH/$DATASET/$DATASET\_groundtruth.ivecs 100 $PATH/$DATASET/$DATASET\_hnsw_$METRIC\_M$DEGREE.index $METRIC $DEGREE

METRIC='ip'
FORMAT=bin
#echo "./bin/faiss_hnsw $FORMAT $PATH/$DATASET/base.$DSIZE.fbin $PATH/$DATASET/query.public.$QSIZE.fbin $PATH/$DATASET/gt100.$DSIZE.ibin 100 $PATH/$DATASET/$DATASET\_hnsw_$METRIC\_M$DEGREE.index $METRIC $DEGREE"
#./bin/faiss_hnsw $FORMAT $PATH/$DATASET/base.$DSIZE.fbin $PATH/$DATASET/query.public.$QSIZE.fbin $PATH/$DATASET/gt100.$DSIZE.ibin 100 $PATH/$DATASET/$DATASET\_hnsw_$METRIC\_M$DEGREE.index $METRIC $DEGREE
