"""Phase 3: HNSW, the hierarchical navigable small world graph.

Every vector is a node. Each node lives on layer 0 and, with geometrically
decreasing probability, on higher layers too. Each layer is a proximity
graph: a node links to a handful of near neighbours chosen to point in
different directions. Search starts at the single entry point on the top
layer, greedily walks to the closest node, drops a layer, repeats, and on
layer 0 widens into a beam search of width `ef_search`.

Pure Python with NumPy for the distance maths. Correctness over speed: the
adjacency lists are plain Python lists and the per-query work is dominated
by interpreter overhead, so compare indexes on `dist_comps` (vectors
touched per query) as well as wall clock.
"""

import heapq
import math
import time

import numpy as np


class HNSWIndex:
    name = "hnsw"

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 100,
        ef_search: int = 50,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        self.M = M                     # max links per node on layers > 0
        self.M0 = 2 * M                # max links per node on layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self.verbose = verbose
        self._mult = 1.0 / math.log(M)  # level ~ floor(-ln(u) * mult)
        self._vecs: np.ndarray | None = None
        self._neighbors: list[list[list[int]]] = []  # [node][layer] -> neighbour ids
        self._entry = -1
        self._max_level = -1
        self._n_comps = 0
        self.last_dist_comps = 0

    # ------------------------------------------------------------ distances

    def _dist(self, q: np.ndarray, ids: list[int]) -> np.ndarray:
        """Cosine distance from q to each id, batched into one matmul."""
        self._n_comps += len(ids)
        return 1.0 - self._vecs[ids] @ q

    # --------------------------------------------------------------- search

    def _search_layer(self, q: np.ndarray, entry: list[int], ef: int, layer: int) -> list[tuple[float, int]]:
        """Beam search on one layer (paper Alg. 2). Returns up to `ef`
        (distance, id) pairs, closest first. ef=1 is a greedy walk."""
        d0 = self._dist(q, entry).tolist()
        visited = set(entry)
        cands = list(zip(d0, entry))            # min-heap on distance
        heapq.heapify(cands)
        results = [(-d, i) for d, i in cands]   # max-heap via negation
        heapq.heapify(results)
        while cands:
            d, c = heapq.heappop(cands)
            if d > -results[0][0]:
                break  # nearest unexplored is already worse than our worst kept
            neigh = [n for n in self._neighbors[c][layer] if n not in visited]
            if not neigh:
                continue
            visited.update(neigh)
            worst = -results[0][0]
            for dn, n in zip(self._dist(q, neigh).tolist(), neigh):
                if len(results) < ef or dn < worst:
                    heapq.heappush(cands, (dn, n))
                    heapq.heappush(results, (-dn, n))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst = -results[0][0]
        return sorted((-d, i) for d, i in results)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._vecs is not None, "call build() first"
        q = query.astype(np.float32)
        self._n_comps = 0
        ep = self._entry
        for layer in range(self._max_level, 0, -1):
            ep = self._search_layer(q, [ep], 1, layer)[0][1]
        res = self._search_layer(q, [ep], max(self.ef_search, k), 0)[:k]
        self.last_dist_comps = self._n_comps
        ids = np.fromiter((i for _, i in res), dtype=np.int64, count=len(res))
        dists = np.fromiter((d for d, _ in res), dtype=np.float32, count=len(res))
        return ids, dists

    # ---------------------------------------------------------------- build

    def _select_neighbors(self, cands: list[tuple[float, int]], M: int) -> list[int]:
        """Paper Alg. 4 heuristic. Walk candidates closest-first and keep one
        only if it is closer to the query than to everything kept so far.
        That spreads links across directions instead of clumping them in one
        dense spot, which is what keeps the graph navigable. Pruned
        candidates backfill if we end up short."""
        selected: list[int] = []
        pruned: list[int] = []
        for d, c in cands:
            if len(selected) >= M:
                break
            if selected:
                d_sel = self._dist(self._vecs[c], selected)
                if d >= d_sel.min():
                    pruned.append(c)
                    continue
            selected.append(c)
        for c in pruned:
            if len(selected) >= M:
                break
            selected.append(c)
        return selected

    def _insert(self, i: int, level: int) -> None:
        q = self._vecs[i]
        self._neighbors.append([[] for _ in range(level + 1)])
        if self._entry < 0:
            self._entry, self._max_level = i, level
            return
        ep = self._entry
        # greedy descent through layers above this node's level
        for layer in range(self._max_level, level, -1):
            ep = self._search_layer(q, [ep], 1, layer)[0][1]
        # on each layer the node belongs to: find candidates, link both ways
        for layer in range(min(level, self._max_level), -1, -1):
            cands = self._search_layer(q, [ep], self.ef_construction, layer)
            m_max = self.M0 if layer == 0 else self.M
            neigh = self._select_neighbors(cands, self.M)
            self._neighbors[i][layer] = neigh
            for n in neigh:
                nl = self._neighbors[n][layer]
                nl.append(i)
                if len(nl) > m_max:
                    dn = self._dist(self._vecs[n], nl).tolist()
                    self._neighbors[n][layer] = self._select_neighbors(sorted(zip(dn, nl)), m_max)
            ep = cands[0][1]
        if level > self._max_level:
            self._entry, self._max_level = i, level

    def build(self, vectors: np.ndarray) -> None:
        self._vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self._neighbors = []
        self._entry, self._max_level = -1, -1
        rng = np.random.default_rng(self.seed)
        n = self._vecs.shape[0]
        levels = np.floor(-np.log(1.0 - rng.random(n)) * self._mult).astype(int)
        t0 = time.perf_counter()
        for i in range(n):
            self._insert(i, int(levels[i]))
            if self.verbose and (i + 1) % 10000 == 0:
                print(f"  hnsw: {i + 1}/{n} inserted, {time.perf_counter() - t0:.0f}s")

    # ---------------------------------------------------------------- misc

    def memory_bytes(self) -> int:
        """Vectors plus adjacency, counted as a compact int32 layout would."""
        if self._vecs is None:
            return 0
        edges = sum(len(l) for node in self._neighbors for l in node)
        return self._vecs.nbytes + 4 * edges

    def params(self) -> str:
        return f"M={self.M} efc={self.ef_construction} ef={self.ef_search}"

    def stats(self) -> dict:
        n = len(self._neighbors)
        per_level = np.bincount([len(x) - 1 for x in self._neighbors])
        deg0 = [len(x[0]) for x in self._neighbors]
        return {
            "nodes": n,
            "max_level": self._max_level,
            "nodes_per_level": per_level.tolist(),
            "mean_degree_layer0": float(np.mean(deg0)),
        }
