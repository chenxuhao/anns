import time
import numpy as np
from sklearn.metrics import pairwise_distances
from numba import cuda
import faiss
import hnswlib
import cuvs
import rmm
from cuvs.neighbors import ivf_flat, cagra, hnsw

# -- Load SIFT1M data
def fvecs_read(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype='int32')
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].astype('float32')

def ivecs_read(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype='int32')
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].astype('int32')

print("Loading SIFT1M dataset...")
base = fvecs_read("sift/sift_base.fvecs")  # 1M x 128
queries = fvecs_read("sift/sift_query.fvecs")  # 10k x 128
gt = ivecs_read("sift/sift_groundtruth.ivecs")  # 10k x 100

print("Data shapes:", base.shape, queries.shape, gt.shape)

k = 10  # number of neighbors to search

# -- Recall@k
def recall_at_k(retrieved, ground_truth, k=10):
    correct = 0
    for r, g in zip(retrieved, ground_truth):
        correct += len(set(r[:k]) & set(g[:k]))
    return correct / (len(ground_truth) * k)

# -- FAISS IVF-Flat (CPU)
def faiss_ivf(base, queries, k=10):
#    dim = base.shape[1]
#    quantizer = faiss.IndexFlatL2(dim)
#    index = faiss.IndexIVFFlat(quantizer, dim, 1024, faiss.METRIC_L2)
#    index.train(base)
#    index.add(base)
#    index.nprobe = 64

    index = faiss.index_factory(128, "IVF100,Flat", faiss.METRIC_L2)
    index.train(base.astype(np.float32))
    index.add(base.astype(np.float32))
    start = time.time()
    distances, neighbors = index.search(queries.astype(np.float32), 10)
#    distances, neighbors = index.search(queries, k)
    search_time = time.time() - start

    return neighbors, distances, search_time

# -- FAISS HNSW (CPU)
def faiss_hnsw(base, queries, k=10):
    dim = base.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efSearch = 64
    index.add(base)

    start = time.time()
    distances, neighbors = index.search(queries, k)
    search_time = time.time() - start

    return neighbors, distances, search_time

# -- cuVS IVF-Flat (GPU)
def cuvs_ivf(base, queries, k=10):
    base_gpu = cuda.to_device(base)
    queries_gpu = cuda.to_device(queries)

    index_params = ivf_flat.IndexParams(n_lists=1024, kmeans_n_iters=20)
    search_params = ivf_flat.SearchParams(n_probes=64)

    index = ivf_flat.build(index_params, base_gpu)

    start = time.time()
    neighbors, distances = ivf_flat.search(search_params, index, queries_gpu, k)
    search_time = time.time() - start

    neighbors_host = neighbors.copy_to_host()
    distances_host = distances.copy_to_host()

    return neighbors_host, distances_host, search_time

def normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, a_min=1e-6, a_max=None)

# -- cuVS HNSW (GPU)
def cuvs_hnsw(base, queries, k=10):
    base = normalize(base)
    queries = normalize(queries)
    base_gpu = cuda.to_device(base)
    queries_gpu = cuda.to_device(queries)

    # Build CAGRA index first (HNSW currently wraps CAGRA in cuVS)
    cagra_index = cagra.build(cagra.IndexParams(), base_gpu)

    # Convert CAGRA to HNSW index
    hnsw_index = hnsw.from_cagra(hnsw.IndexParams(), cagra_index)

    # Search
    search_params = hnsw.SearchParams(ef=64)

    start = time.time()
    neighbors, distances = hnsw.search(search_params, hnsw_index, queries.astype(np.float32), k)
    search_time = time.time() - start

    return neighbors, distances, search_time

# -- HNSWlib (CPU)
def hnswlib_ann(base, queries, k=10):
    dim = base.shape[1]
    num_elements = base.shape[0]

    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=200, M=32)
    p.add_items(base)
    p.set_ef(64)

    start = time.time()
    neighbors, distances = p.knn_query(queries, k=k)
    search_time = time.time() - start

    return neighbors, distances, search_time

# -- Run benchmarks
methods = {
    "FAISS IVF-Flat (CPU)": faiss_ivf,
#    "FAISS HNSW (CPU)": faiss_hnsw,
#    "cuVS IVF-Flat (GPU)": cuvs_ivf,
#    "cuVS HNSW (GPU)": cuvs_hnsw,
#    "HNSWlib (CPU)": hnswlib_ann
}

results = {}

for name, method in methods.items():
    print(f"\nRunning {name}...")
    neighbors, distances, search_time = method(base, queries, k)
    recall = recall_at_k(neighbors, gt, k)
    results[name] = {
        "search_time": search_time,
        "recall": recall
    }

# -- Display results
print("\nBenchmark Results:")
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  Search Time: {metrics['search_time']:.3f} s")
    print(f"  Recall@{k}: {metrics['recall']:.4f}")

