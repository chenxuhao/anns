# ANN Search
[This challenge](https://big-ann-benchmarks.com/neurips23.html) is to encourage the development of indexing data structures and search algorithms 
for practical variants of the Approximate Nearest Neighbor (ANN) or Vector search problem. 

## Quick Start

Edit `env.sh` to let the libraries pointing to the right paths in your system, and then:

    $ source env.sh

Unpack the `siftsmall` dataset:

    $ cd data; tar zxvf siftsmall.tar.gz

Now build the program:

    $ make

Binaries wiil be generated in the `bin` directory.

To run the brute-force implementation:

    $ cd bin
    $ ./test_brute.sh

To run the quantized implementation:

    $ ./test_quantized.sh


## References

[big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks/tree/main)

[ann-benchmarks](https://ann-benchmarks.com/)

[Billion-Scale Similarity Search with GPUs](https://arxiv.org/abs/1702.08734) [BigData] [Code](https://github.com/facebookresearch/faiss)

[DiskANN](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) 
[Video](https://www.microsoft.com/en-us/research/video/research-talk-approximate-nearest-neighbor-search-systems-at-scale/)
[Slides](https://cvpr.thecvf.com/media/cvpr-2023/Slides/18545_SzZdLZD.pdf)
[Slides 2](https://people.csail.mit.edu/jshun/6506-s24/lectures/lecture21-2.pdf)

[Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf)

[ParlayANN: Scalable and Deterministic Parallel Graph Based Approximate Nearest Neighbor Search Algorithms](https://arxiv.org/pdf/2305.04359.pdf)

[Co-design Hardware and Algorithm for Vector Search](https://dl.acm.org/doi/pdf/10.1145/3581784.3607045)

[Worst-case Performance of Popular Approximate Nearest Neighbor Search Implementations: Guarantees and Limitations](https://proceedings.neurips.cc/paper_files/paper/2023/file/d0ac28b79816b51124fcc804b2496a36-Paper-Conference.pdf)

## More Datasets

[DEEP1B](https://www.tensorflow.org/datasets/catalog/deep1b) -- iQAN randomly sampled 10m and 100m

[SIFT1B](https://www.tensorflow.org/datasets/catalog/sift1m) -- iQAN randomly sampled 100m, and also used the 1m version

[GIST1M](http://corpus-texmex.irisa.fr/)
