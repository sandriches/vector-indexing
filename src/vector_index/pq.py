"""Phase 4: product quantization.

Split every d-dim vector into `m` subvectors of d/m dims. Run k-means with
256 centroids independently in each subspace. A vector is then stored as m
uint8 codes (one centroid id per subspace): 384 float32 = 1536 bytes becomes
m bytes.

Search uses asymmetric distance computation (ADC): the query stays exact,
and for each subspace we precompute the inner product of the query's
subvector with all 256 centroids. The distance to any stored vector is then
m table lookups and a sum, never touching the original floats.

Two indexes live here:
- PQIndex: flat scan of every code (memory story, not a speed story).
- IVFPQIndex: IVF coarse quantizer, PQ trained on the residual (x - centroid)
  inside each cell. Residuals are much smaller than raw vectors so the same
  number of bytes gives a better approximation. This is what FAISS's
  IVFxx,PQyy does.
Both take an optional `rerank`: re-score the top `rerank` ADC candidates
with the exact vectors to win recall back.
"""

import numpy as np

from .kmeans import assign, kmeans


class ProductQuantizer:
    def __init__(self, m: int = 16, ksub: int = 256, n_iter: int = 20, seed: int = 0) -> None:
        self.m = m
        self.ksub = ksub
        self.n_iter = n_iter
        self.seed = seed
        self.dsub = 0
        self.codebooks: np.ndarray | None = None  # (m, ksub, dsub)

    def train(self, X: np.ndarray) -> None:
        n, d = X.shape
        if d % self.m:
            raise ValueError(f"d={d} not divisible by m={self.m}")
        self.dsub = d // self.m
        self.codebooks = np.empty((self.m, self.ksub, self.dsub), dtype=np.float32)
        for j in range(self.m):
            sub = X[:, j * self.dsub : (j + 1) * self.dsub]
            self.codebooks[j], _ = kmeans(sub, self.ksub, n_iter=self.n_iter, seed=self.seed + j)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """(n, d) float -> (n, m) uint8 codes."""
        codes = np.empty((X.shape[0], self.m), dtype=np.uint8)
        for j in range(self.m):
            sub = np.ascontiguousarray(X[:, j * self.dsub : (j + 1) * self.dsub])
            codes[:, j], _ = assign(sub, self.codebooks[j])
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """(n, m) codes -> (n, d) reconstructed float32."""
        out = np.empty((codes.shape[0], self.m * self.dsub), dtype=np.float32)
        for j in range(self.m):
            out[:, j * self.dsub : (j + 1) * self.dsub] = self.codebooks[j][codes[:, j]]
        return out

    def ip_tables(self, q: np.ndarray) -> np.ndarray:
        """(m, ksub) table: inner product of each query subvector with every
        centroid in that subspace. q may be (d,) or (p, d) -> (p, m, ksub)."""
        qs = q.reshape(*q.shape[:-1], self.m, self.dsub)
        return np.einsum("mkd,...md->...mk", self.codebooks, qs)

    def memory_bytes(self) -> int:
        return 0 if self.codebooks is None else self.codebooks.nbytes


def _topk(dists: np.ndarray, k: int) -> np.ndarray:
    k = min(k, dists.size)
    part = np.argpartition(dists, k - 1)[:k]
    return part[np.argsort(dists[part])]


class PQIndex:
    name = "pq"

    def __init__(self, m: int = 16, rerank: int = 0, train_size: int = 20000, seed: int = 0) -> None:
        self.m = m
        self.rerank = rerank
        self.train_size = train_size
        self.seed = seed
        self.pq = ProductQuantizer(m=m, seed=seed)
        self._codes_T: np.ndarray | None = None  # (m, n) so each subspace's column is contiguous
        self._vecs: np.ndarray | None = None     # originals, only needed for rerank
        self.last_dist_comps = 0

    def build(self, vectors: np.ndarray) -> None:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        train = X if self.train_size >= X.shape[0] else X[rng.choice(X.shape[0], self.train_size, replace=False)]
        self.pq.train(train)
        self._codes_T = np.ascontiguousarray(self.pq.encode(X).T)
        self._vecs = X

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = query.astype(np.float32)
        T = self.pq.ip_tables(q)                       # (m, 256)
        n = self._codes_T.shape[1]
        ip = np.zeros(n, dtype=np.float32)
        for j in range(self.m):                        # m gathers of n bytes each
            ip += T[j][self._codes_T[j]]
        dists = 1.0 - ip
        self.last_dist_comps = n
        if self.rerank > k:
            cand = _topk(dists, self.rerank)
            exact = 1.0 - self._vecs[cand] @ q
            self.last_dist_comps += cand.size
            top = _topk(exact, k)
            return cand[top], exact[top]
        top = _topk(dists, k)
        return top, dists[top]

    def memory_bytes(self) -> int:
        if self._codes_T is None:
            return 0
        b = self._codes_T.nbytes + self.pq.memory_bytes()
        if self.rerank:
            b += self._vecs.nbytes  # exact vectors must be kept somewhere
        return b

    def params(self) -> str:
        return f"m={self.m} rerank={self.rerank}"


class IVFPQIndex:
    name = "ivfpq"

    def __init__(
        self,
        nlist: int = 1024,
        m: int = 16,
        nprobe: int = 16,
        rerank: int = 0,
        train_size: int = 20000,
        seed: int = 0,
    ) -> None:
        self.nlist = nlist
        self.m = m
        self.nprobe = nprobe
        self.rerank = rerank
        self.train_size = train_size
        self.seed = seed
        self.pq = ProductQuantizer(m=m, seed=seed)
        self.centroids: np.ndarray | None = None
        self._codes: np.ndarray | None = None    # (n, m) sorted by cluster
        self._ids: np.ndarray | None = None
        self._offsets: np.ndarray | None = None
        self._vecs: np.ndarray | None = None     # originals, only for rerank
        self.last_dist_comps = 0
        self.last_scanned = 0

    def build(self, vectors: np.ndarray) -> None:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        sample = rng.choice(n, min(self.train_size, n), replace=False)
        self.centroids, _ = kmeans(X[sample], self.nlist, seed=self.seed)
        labels, _ = assign(X, self.centroids)
        residuals = X - self.centroids[labels]
        self.pq.train(residuals[sample])
        codes = self.pq.encode(residuals)
        order = np.argsort(labels, kind="stable")
        self._ids = order.astype(np.int64)
        self._codes = np.ascontiguousarray(codes[order])
        counts = np.bincount(labels, minlength=self.nlist)
        self._offsets = np.concatenate([[0], np.cumsum(counts)])
        self._vecs = X

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = query.astype(np.float32)
        c_ip = self.centroids @ q
        nprobe = min(self.nprobe, self.nlist)
        probe = np.argpartition(-c_ip, nprobe - 1)[:nprobe]
        # rows in the probed cells, and which probe each row came from
        starts, stops = self._offsets[probe], self._offsets[probe + 1]
        lens = stops - starts
        rows = np.concatenate([np.arange(s, e) for s, e in zip(starts, stops)])
        which = np.repeat(np.arange(nprobe), lens)
        self.last_scanned = rows.size
        self.last_dist_comps = rows.size + self.nlist
        if rows.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        # q.x = q.c + q.(x - c): first term per cell, second via ADC tables
        # built from the query's residual against each probed centroid.
        T = self.pq.ip_tables(q - self.centroids[probe])          # (nprobe, m, 256)
        codes = self._codes[rows]                                  # (R, m)
        res_ip = T[which[:, None], np.arange(self.m)[None, :], codes].sum(axis=1)
        dists = 1.0 - (c_ip[probe][which] + res_ip)
        if self.rerank > k:
            cand = _topk(dists, self.rerank)
            ids = self._ids[rows[cand]]
            exact = 1.0 - self._vecs[ids] @ q
            self.last_dist_comps += cand.size
            top = _topk(exact, k)
            return ids[top], exact[top]
        top = _topk(dists, k)
        return self._ids[rows[top]], dists[top]

    def memory_bytes(self) -> int:
        if self._codes is None:
            return 0
        b = self._codes.nbytes + self.pq.memory_bytes() + self.centroids.nbytes + self._ids.nbytes + self._offsets.nbytes
        if self.rerank:
            b += self._vecs.nbytes
        return b

    def params(self) -> str:
        return f"nlist={self.nlist} m={self.m} nprobe={self.nprobe} rerank={self.rerank}"
