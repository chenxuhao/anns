import numpy as np
import faiss
import matplotlib.pyplot as plt
import os
import argparse
from collections import Counter
import time

def read_fvecs(fname):
    """Reads a .fvecs file into a float32 numpy array of shape (n, d)"""
    a = np.fromfile(fname, dtype='int32')
    if a.size == 0:
        raise ValueError(f"Empty file: {fname}")
    d = a[0]
    assert a.size % (d + 1) == 0, f"Corrupt .fvecs file: {fname}"
    return a.reshape(-1, d + 1)[:, 1:].view('float32')

def read_ivecs(filename):
    """Reads a .ivecs file into an int32 numpy array of shape (n, d)"""
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

def run_ivf_histogram(xb, xq, gt, nprobe, topk):
    nb, dim = xb.shape
    nq = xq.shape[0]
    print(f"Base vectors: {nb}, dim: {dim}")
    print(f"Query vectors: {nq}, topk: {topk}")
    print(f"Groundtruth shape: {gt.shape}")

    # Choose nlist heuristically
    nlist = int(2 ** round(np.log2(np.sqrt(nb))))
    print(f"Using IVF index with nlist={nlist}, nprobe={nprobe}")

    # Build index
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    train_size = min(100000, xb.shape[0])
    print("training size:", train_size)
    train_x = xb[np.random.choice(xb.shape[0], train_size, replace=False)]
    index.train(train_x)
    print("is_trained:", index.is_trained)
    index.add(xb)
    index.nprobe = nprobe

    # Search
    start = time.time()
    D, I = index.search(xq, topk)
    end = time.time()

    recall = compute_recall(I, gt, topk=topk, gt_k=gt.shape[1])
    total_time_ms = (end - start) * 1000
    throughput = nq / (end - start)
    avg_latency = total_time_ms / nq

    print(f"\nRecall@{topk}: {recall:.4f}")
    print(f"==== Search Statistics ====")
    print(f"nprobe: {nprobe}")
    print(f"Total time: {total_time_ms:.2f} ms")
    print(f"Avg latency: {avg_latency:.4f} ms/query")
    print(f"Throughput: {throughput:.2f} queries/sec")

    # Get cluster assignments
    _, cluster_ids = index.quantizer.search(xq, nprobe)
    cluster_freq = Counter(cluster_ids.flatten())
    all_counts = np.array([cluster_freq[cid] for cid in range(nlist)])

    # Compute total number of cluster accesses
    total_accesses = np.sum(all_counts)

    # Sort cluster access frequencies in descending order
    sorted_indices = np.argsort(-all_counts)  # Indices of clusters sorted by frequency

    top_k = int(0.2 * nlist)  # Top 10% clusters
    top_clusters = sorted_indices[:top_k]
    top_accesses = np.sum(all_counts[top_clusters])

    # Fraction of total accesses from top 10% clusters
    access_coverage_ratio = top_accesses / total_accesses
    print(f"Top 20% clusters cover {access_coverage_ratio:.4f} of total accesses.")

    # Estimate storage required for top 10% clusters
    # Count how many database vectors fall into each cluster
    _, xb_assignments = index.quantizer.search(xb, 1)
    xb_assignments = xb_assignments.flatten()
    xb_cluster_counts = Counter(xb_assignments)
    top_cluster_storage = sum(xb_cluster_counts[cid] for cid in top_clusters)

    # Size per vector (in bytes)
    vector_bytes = xb.shape[1] * 4  # float32 = 4 bytes

    # Total bytes for top 10% clusters
    top_cluster_bytes = top_cluster_storage * vector_bytes

    # Convert to MB and GB
    top_cluster_MB = top_cluster_bytes / (1024 ** 2)
    top_cluster_GB = top_cluster_bytes / (1024 ** 3)

    print(f"Top 20% clusters hold {top_cluster_storage} vectors.")
    print(f"Top 20% clusters storage: {top_cluster_MB:.2f} MB ({top_cluster_GB:.2f} GB)")

    # Plot
    plt.figure(figsize=(10, 5))
    sorted_freq = np.sort(all_counts)[::-1]
    plt.plot(sorted_freq, label='Cluster Access Frequency')
    plt.xlabel("Cluster Rank (sorted)")
    plt.ylabel("# of Query Assignments")
    title = f"{os.path.basename(data_prefix)} - IVF Histogram (nlist={nlist}, nprobe={nprobe})"
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    out_file = f"{os.path.basename(data_prefix)}_ivf_hist_nlist{nlist}_nprobe{nprobe}.png"
    plt.savefig(out_file)
    plt.show()

# === Entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["sift1M", "deep1M", "sift10M", "gist1M"])
    parser.add_argument("--nprobe", type=int, default=32, help="Number of clusters to probe")
    parser.add_argument("--topk", type=int, default=100, help="top k")
    args = parser.parse_args()

    dataset_paths = {
        "sift1M": "data/sift1M/sift",
        "gist1M": "data/gist1M/gist",
        "deep1M": "data/deep1M/deep1M",
        "sift10M": "data/sift10M/sift10M"
    }
    data_prefix = dataset_paths[args.dataset]
    base_file = data_prefix + "_base.fvecs"
    query_file = data_prefix + "_query.fvecs"
    gt_file = data_prefix + "_groundtruth.ivecs"
    # print(base_file)
    # print(query_file)
    # print(gt_file)

    xb = read_fvecs(base_file).astype(np.float32)
    xq = read_fvecs(query_file).astype(np.float32)
    gt = read_ivecs(gt_file)

    run_ivf_histogram(xb, xq, gt, args.nprobe, args.topk)
