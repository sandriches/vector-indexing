"""Phase 1: exact nearest-neighbour search by scanning everything.

Not an index, but the baseline every index is measured against, and the
source of ground truth for recall.
"""

import numpy as np


class BruteForceIndex:
    name = "brute"

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        self._vectors = np.ascontiguousarray(vectors)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        ids, dists = self.search_batch(query[None, :], k)
        return ids[0], dists[0]

    def search_batch(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search many queries at once. Returns (ids, dists) of shape (n_queries, k)."""
        assert self._vectors is not None, "call build() first"
        # One matrix multiply gives every similarity. Rows are normalised, so
        # dot product == cosine similarity.
        sims = queries.astype(np.float32) @ self._vectors.T
        dists = 1.0 - sims
        # argpartition finds the k smallest in O(n) without sorting all n.
        # Only the k survivors then get a full sort.
        k = min(k, dists.shape[1])
        part = np.argpartition(dists, k - 1, axis=1)[:, :k]
        part_d = np.take_along_axis(dists, part, axis=1)
        order = np.argsort(part_d, axis=1)
        ids = np.take_along_axis(part, order, axis=1)
        out_d = np.take_along_axis(part_d, order, axis=1)
        return ids, out_d

    def memory_bytes(self) -> int:
        return 0 if self._vectors is None else self._vectors.nbytes
