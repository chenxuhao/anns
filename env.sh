
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

module purge
module load anaconda/2021.05-py38
module load modtree/gpu
module load python
module load gcc/11.2.0
module load cmake/3.20.0
module load cuda/11.2.2
module list

export LD_LIBRARY_PATH=$HOME/proj/faiss/build/faiss:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/proj/amd-libm/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/proj/lapack-3.10.1/:$LD_LIBRARY_PATH