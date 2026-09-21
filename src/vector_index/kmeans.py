"""k-means in NumPy: k-means++ seeding, then Lloyd's algorithm.

Used by IVF (Phase 2) for the coarse quantizer and later by PQ (Phase 4)
for the codebooks. Distances are squared L2 throughout, computed as
|x|^2 - 2 x.c + |c|^2 so the heavy part is a single matmul.
"""

import numpy as np


def assign(X: np.ndarray, centroids: np.ndarray, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Nearest centroid for every row of X. Returns (labels, squared distances).

    Chunked so the (n, k) distance matrix never has to exist all at once.
    """
    c_norm = (centroids * centroids).sum(axis=1)
    labels = np.empty(X.shape[0], dtype=np.int64)
    dists = np.empty(X.shape[0], dtype=np.float32)
    for i in range(0, X.shape[0], chunk):
        xb = X[i : i + chunk]
        d = (xb * xb).sum(axis=1)[:, None] - 2.0 * (xb @ centroids.T) + c_norm[None, :]
        lab = d.argmin(axis=1)
        labels[i : i + chunk] = lab
        dists[i : i + chunk] = np.maximum(d[np.arange(len(lab)), lab], 0.0)
    return labels, dists


def kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++: pick each new centre with probability proportional to its
    squared distance from the nearest centre chosen so far."""
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=X.dtype)
    centroids[0] = X[rng.integers(n)]
    x_norm = (X * X).sum(axis=1)
    # running min squared distance to the chosen set
    best = x_norm - 2.0 * (X @ centroids[0]) + centroids[0] @ centroids[0]
    best = np.maximum(best, 0.0)
    for j in range(1, k):
        probs = best / best.sum() if best.sum() > 0 else np.full(n, 1.0 / n)
        idx = rng.choice(n, p=probs)
        centroids[j] = X[idx]
        d = x_norm - 2.0 * (X @ centroids[j]) + centroids[j] @ centroids[j]
        best = np.minimum(best, np.maximum(d, 0.0))
    return centroids


def kmeans(
    X: np.ndarray,
    k: int,
    n_iter: int = 20,
    seed: int = 0,
    tol: float = 1e-4,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm. Returns (centroids (k, d), labels (n,)).

    Empty clusters are re-seeded from the points furthest from their
    current centre, so we always end with exactly k live centroids.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if k > X.shape[0]:
        raise ValueError(f"k={k} but only {X.shape[0]} points")
    centroids = kmeans_pp_init(X, k, rng)
    prev_inertia = np.inf
    labels = np.zeros(X.shape[0], dtype=np.int64)
    for it in range(n_iter):
        labels, d = assign(X, centroids)
        inertia = float(d.sum())
        counts = np.bincount(labels, minlength=k)
        # sum rows per cluster, then divide
        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, X)
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
        empty = np.flatnonzero(~nonempty)
        if empty.size:
            far = np.argsort(d)[-empty.size :]
            centroids[empty] = X[far]
        if verbose:
            print(f"iter {it:3d} inertia {inertia:.4f} empty {empty.size}")
        if prev_inertia - inertia < tol * max(prev_inertia, 1e-12) and not empty.size:
            break
        prev_inertia = inertia
    labels, _ = assign(X, centroids)
    return centroids, labels
