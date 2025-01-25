import pyanns
import numpy as np
import gc
import time
from concurrent.futures import ThreadPoolExecutor
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
        num = int.from_bytes(f.read(4), byteorder='little')
        print(f"n = {num}, dim = {dim}\t", end="")
        total_size = num * dim * np.dtype(np.float32).itemsize
        
        f.seek(8)  # Skip header
        vectors = np.frombuffer(f.read(total_size), dtype=np.float32).reshape(num, dim)
        # Ensure memory alignment
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    return num, vectors
  
def load_gt(filename, dim):
    with open(filename, 'rb') as f:
        num = int.from_bytes(f.read(4), byteorder='little')
        # The total size should account for both num and dim
        total_size = num * dim * np.dtype(np.int32).itemsize
        
        f.seek(8)  # Skip header
        vectors = np.frombuffer(f.read(total_size), dtype=np.int32).reshape(num, dim)
        vectors = np.ascontiguousarray(vectors, dtype=np.int32)
    return vectors
  
def perform_search(searcher, queries, k):
    return searcher.batch_search(queries, k)

if __name__ == "__main__":
    d = 512
    k = 100
    
    # build diskann index
    graph_path = "/work2/10442/ychoi7/stampede3/data/diskann_index/laion-10M/laion10m-index2.2"

    g = pyanns.Graph(graph_path, 'diskann')
    print('Graph loaded')
    
    _, data = load_vectors_fbin("/work2/10442/ychoi7/stampede3/data/laion-10M/base.10M.fbin", d)
    
    print('Data loaded')
    searcher = pyanns.Searcher(
        g, data, "mips", "SQ8U")
    
    print('Searcher created')
    searcher.optimize()
    
    print('Index ready for search')
    
    _, query_data = load_vectors_fbin("/work2/10442/ychoi7/stampede3/data/laion-10M/query.10k.fbin", d)
    nq, _ = query_data.shape
    
    query_chunks = np.array_split(query_data, 128)
    results = []

    t0 = time.time()
    
    # Use ThreadPoolExecutor for parallel search
    with ThreadPoolExecutor(max_workers=128) as executor:
        future_results = [executor.submit(perform_search, searcher, chunk, k) for chunk in query_chunks]
        for future in future_results:
            results.extend(future.result())
    
    results = np.array(results).reshape(nq, -1)
    print('Search done')
    
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.2f} seconds")
    print(f"Queries per second: {nq / (t1 - t0):.2f}")
    
    
    gt_path = "/work2/10442/ychoi7/stampede3/data/diskann_index/laion-10M/laion10m-gt100"
    # gt_path = "/work2/10442/ychoi7/stampede3/data/laion-10M/gt.10k.ibin"
    
    # read gt
    gt = load_gt(gt_path, 100)
    
    # calculate recall
    # recall = (res == gt).sum() / nq
    # print(f"Recall: {recall:.4f}")

    missing_rate = (results == -1).sum() / float(k * nq)

    correct = 0
    for i in range(nq):
        correct += np.intersect1d(results[i], gt[i]).size
    
    recall = correct / float(k * nq)
    print(f"Recall: {recall:.4f}")
    print(f"Missing rate: {missing_rate:.4f}")