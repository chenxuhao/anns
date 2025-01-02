import pyanns
import numpy as np
import gc

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
  
if __name__ == "__main__":
    d = 200
    k = 100
    # build diskann index
    graph_path = "/home/x-ychoi7/DiskANN/build/data/t2i10m-index.2"
    
    # gc.collect()
    g = pyanns.Graph(graph_path, 'diskann')
    
    data = load_vectors_fbin("/home/x-ychoi7/newproj/anns/ann_data/t2i-10M/base.10M.fbin", d)
    # data = np.ascontiguousarray(data)
    
    searcher = pyanns.Searcher(
        g, data, "IP", "SQ8U")
    searcher.optimize()
    
    print('Index ready for search')
    
    query_data = load_vectors_fbin("/home/x-ychoi7/newproj/anns/ann_data/t2i-10M/query.1000.fbin", d)
    # query_data = np.ascontiguousarray(query_data)
    nq, _ = query_data.shape
    res = searcher.batch_search(query_data, k).reshape(nq, -1)
    
