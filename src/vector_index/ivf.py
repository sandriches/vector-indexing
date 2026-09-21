"""Phase 2: IVF, the inverted file index.

Build: run k-means to get `nlist` centroids, assign every vector to its
nearest centroid, and store the vectors grouped by centroid (CSR-style, so
each cluster is one contiguous block).

Search: find the `nprobe` centroids closest to the query, then brute-force
only the vectors in those clusters. `nprobe` is the recall/speed knob.
"""

import numpy as np

from .kmeans import assign, kmeans


class IVFIndex:
    name = "ivf"

    def __init__(
        self,
        nlist: int = 1024,
        nprobe: int = 8,
        n_iter: int = 20,
        train_size: int | None = None,
        seed: int = 0,
    ) -> None:
        self.nlist = nlist
        self.nprobe = nprobe
        self.n_iter = n_iter
        self.train_size = train_size  # subsample for k-means; None = use all
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self._vecs: np.ndarray | None = None      # vectors sorted by cluster
        self._ids: np.ndarray | None = None       # original id of each sorted row
        self._offsets: np.ndarray | None = None   # cluster c is rows offsets[c]:offsets[c+1]
        self.last_scanned = 0                     # vectors compared on the last search

    def build(self, vectors: np.ndarray) -> None:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        train = X
        if self.train_size is not None and self.train_size < X.shape[0]:
            train = X[rng.choice(X.shape[0], self.train_size, replace=False)]
        self.centroids, _ = kmeans(train, self.nlist, n_iter=self.n_iter, seed=self.seed)
        labels, _ = assign(X, self.centroids)
        order = np.argsort(labels, kind="stable")
        self._ids = order.astype(np.int64)
        self._vecs = np.ascontiguousarray(X[order])
        counts = np.bincount(labels, minlength=self.nlist)
        self._offsets = np.concatenate([[0], np.cumsum(counts)])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.centroids is not None, "call build() first"
        q = query.astype(np.float32)
        # 1. coarse step: which clusters to look in
        c_sims = self.centroids @ q
        nprobe = min(self.nprobe, self.nlist)
        probe = np.argpartition(-c_sims, nprobe - 1)[:nprobe]
        # 2. gather candidate rows from those clusters
        slices = [slice(self._offsets[c], self._offsets[c + 1]) for c in probe]
        rows = np.concatenate([np.arange(s.start, s.stop) for s in slices])
        self.last_scanned = rows.size
        if rows.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        # 3. fine step: exact distances within the candidates
        dists = 1.0 - self._vecs[rows] @ q
        k = min(k, dists.size)
        part = np.argpartition(dists, k - 1)[:k]
        order = np.argsort(dists[part])
        top = part[order]
        return self._ids[rows[top]], dists[top]

    def memory_bytes(self) -> int:
        if self._vecs is None:
            return 0
        return self.centroids.nbytes + self._vecs.nbytes + self._ids.nbytes + self._offsets.nbytes

    def params(self) -> str:
        return f"nlist={self.nlist} nprobe={self.nprobe}"
