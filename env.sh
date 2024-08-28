
export OMP_NUM_THREADS=32

export PROJ_HOME=$HOME/proj

echo $PROJ_HOME
export CUDA_HOME=/usr/local/cuda
export CILK_HOME=$HOME/OpenCilk/build
export CILK_CLANG=$HOME/OpenCilk/build/lib/clang/14.0.6
export FAISS_HOME=$PROJ_HOME/faiss
export RAFT_HOME=$PROJ_HOME/raft/cpp
export OPENBLAS_HOME=$PROJ_HOME/openblas

export ANN_HOME=$PROJ_HOME/anns
export ANN_DATASET_PATH=$PROJ_HOME/anns/data

