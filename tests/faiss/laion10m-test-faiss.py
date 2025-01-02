import faiss
import numpy as np
import time

import numpy as np
import mmap
import os
import numpy as np

def load_vectors_fbin(filename, dim):
    """
    Load vectors from a binary file with a given dimension.
    
    Args:
        filename (str): Path to the binary file.
        dim (int): Dimension of each vector.
        
    Returns:
        tuple: A tuple containing:
            - num (int): Number of vectors.
            - vectors (np.ndarray): Loaded vectors as a NumPy array.
    """
    with open(filename, 'rb') as f:
        # Read the number of vectors from the file
        num = int.from_bytes(f.read(4), byteorder='little')

        print(f"n = {num}, dim = {dim}\t", end="")

        # Calculate total memory size
        total_size = num * dim * np.dtype(np.float32).itemsize

        # Allocate memory for the data
        if (dim * np.dtype(np.float32).itemsize) % 32 == 0:
            print("aligned")
        else:
            print("unaligned")

        # Read and reshape the binary data
        f.seek(8)  # Skip the first 8 bytes (num and dim placeholder)
        vectors = np.frombuffer(f.read(total_size), dtype=np.float32).reshape(num, dim)

    return num, vectors


if __name__ == "__main__":

  base_num, base_data = load_vectors_fbin(f"/home/x-ychoi7/newproj/anns/ann_data/laion-10M/base.10M.fbin", 512)
  query_num, query_data = load_vectors_fbin(f"/home/x-ychoi7/newproj/anns/ann_data/laion-10M/query.10k.fbin", 512)

  dim = 512
  k = 100
  # print("Building exact search index...")
  index_gt = faiss.IndexFlatIP(dim)
  index_gt.add(base_data)
  
  faiss.omp_set_num_threads(128)
  # Perform exact search
  print("Performing exact search...")
  t0 = time.time()
  D_gt, I_gt = index_gt.search(query_data, k)
  t1 = time.time()
  
  print(f"Exact search completed in {t1 - t0:.2f} seconds")
  
  
  t0 = time.time()
  quantizer = faiss.IndexFlatIP(dim)
  index = faiss.IndexIVFFlat(quantizer, dim, 100, faiss.METRIC_INNER_PRODUCT)   # build the index

  assert not index.is_trained
  index.train(base_data)
  assert index.is_trained

  index.add(base_data)
  print(index.ntotal)
  t1 = time.time()
  print(f"Index training completed in {t1 - t0:.2f} seconds")
  
  index_size_bytes = (
      # Size of the quantizer (IndexFlatIP)
      dim * index.nlist * np.dtype(np.float32).itemsize +
      # Size of the IVF data structure
      base_num * (dim * np.dtype(np.float32).itemsize + np.dtype(np.int64).itemsize)
  )
  print(f"Estimated index size: {index_size_bytes / (1024 * 1024):.2f} MB")


  # faiss.omp_set_num_threads(128)
  t0 = time.time()
  k = 100
  index.nprobe = 10
  D, I = index.search(query_data, k)     # actual search
  
  t1 = time.time()
  missing_rate = (I == -1).sum() / float(k * query_data.shape[0])

  correct = 0
  for i in range(query_num):
      correct += np.intersect1d(I[i], I_gt[i]).size
      # correct += np.intersect1d(I[i], gt_data[i]).size
    
  recall = correct / float(k * query_num)

  print("\t %7.3f queries per second, R@%d %.4f, missing rate %.4f" % (
      query_data.shape[0] / (t1 - t0), k, recall, missing_rate))