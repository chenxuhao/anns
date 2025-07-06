import numpy as np
import faiss
import time
import argparse

def read_fvecs(fname):
    a = np.fromfile(fname, dtype='int32')
    if a.size == 0:
        raise ValueError(f"Empty file: {fname}")
    d = a[0]
    assert a.size % (d + 1) == 0, f"Corrupt .fvecs file: {fname}"
    return a.reshape(-1, d + 1)[:, 1:].view('float32')

def read_ivecs(filename):
    a = np.fromfile(filename, dtype='int32')
    if a.size == 0:
        raise ValueError(f"Empty file: {filename}")
    d = a[0]
    assert a.size % (d + 1) == 0, f"File {filename} has incorrect format"
    return a.reshape(-1, d + 1)[:, 1:]

def compute_recall(predicted, groundtruth, topk, gt_k):
    nq = predicted.shape[0]
    correct = 0
    for i in range(nq):
        gt_set = set(groundtruth[i, :gt_k])
        correct += sum(1 for pred in predicted[i, :topk] if pred in gt_set)
    return correct / (nq * topk)

def run_hnsw_search(xb, xq, gt, M, efConstruction, efSearch, topk):
    nb, dim = xb.shape
    nq = xq.shape[0]
    print(f"Base vectors: {nb}, dim: {dim}")
    print(f"Query vectors: {nq}, topk: {topk}")
    print(f"Groundtruth shape: {gt.shape}")

    start = time.time()
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = efConstruction
    index.add(xb)
    index.hnsw.efSearch = efSearch
    end = time.time()
    gc_time = (end - start)
    print(f"Graph construction time: {gc_time:.2f} s")

    start = time.time()
    D, I = index.search(xq, topk)
    end = time.time()

    recall = compute_recall(I, gt, topk=topk, gt_k=gt.shape[1])
    total_time_ms = (end - start) * 1000
    throughput = nq / (end - start)
    avg_latency = total_time_ms / nq

    print(f"\nRecall@{topk}: {recall:.4f}")
    print(f"==== HNSW Search Statistics ====")
    print(f"M: {M}, efConstruction: {efConstruction}, efSearch: {efSearch}")
    print(f"Search time: {total_time_ms:.2f} ms")
    print(f"Avg latency: {avg_latency:.4f} ms/query")
    print(f"Throughput: {throughput:.2f} queries/sec")

# === Entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["sift1M", "deep1M", "sift10M"])
    parser.add_argument("--M", type=int, default=32, help="HNSW graph degree")
    parser.add_argument("--efConstruction", type=int, default=40, help="efConstruction parameter")
    parser.add_argument("--efSearch", type=int, default=64, help="efSearch parameter")
    parser.add_argument("--topk", type=int, default=100, help="top k")
    args = parser.parse_args()

    dataset_paths = {
        "sift1M": "data/sift1M/sift",
        "deep1M": "data/deep1M/deep1M",
        "sift10M": "data/sift10M/sift10m"
    }
    data_prefix = dataset_paths[args.dataset]
    base_file = data_prefix + "_base.fvecs"
    query_file = data_prefix + "_query.fvecs"
    gt_file = data_prefix + "_groundtruth.ivecs"

    xb = read_fvecs(base_file)
    xq = read_fvecs(query_file)
    gt = read_ivecs(gt_file)

    run_hnsw_search(xb, xq, gt, args.M, args.efConstruction, args.efSearch, args.topk)

