from collections import deque

import numpy as np

from vector_index.brute import BruteForceIndex
from vector_index.hnsw import HNSWIndex


def _normed(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _build(n=600, d=16, seed=0, **kw):
    rng = np.random.default_rng(seed)
    base, queries = _normed(rng, n, d), _normed(rng, 25, d)
    idx = HNSWIndex(M=8, ef_construction=50, ef_search=30, seed=seed, verbose=False, **kw)
    idx.build(base)
    return base, queries, idx


def test_exhaustive_ef_is_exact():
    base, queries, idx = _build()
    bf = BruteForceIndex(); bf.build(base)
    idx.ef_search = base.shape[0]
    for q in queries:
        a, da = bf.search(q, 10)
        b, db = idx.search(q, 10)
        assert a.tolist() == b.tolist()
        np.testing.assert_allclose(da, db, atol=1e-5)


def test_default_ef_has_high_recall():
    base, queries, idx = _build()
    bf = BruteForceIndex(); bf.build(base)
    hits = 0
    for q in queries:
        a, _ = bf.search(q, 10)
        b, _ = idx.search(q, 10)
        hits += len(set(a.tolist()) & set(b.tolist()))
    assert hits / (len(queries) * 10) >= 0.9
    assert idx.last_dist_comps < base.shape[0]


def test_results_sorted_distinct_and_true_distances():
    base, queries, idx = _build()
    for q in queries:
        ids, d = idx.search(q, 7)
        assert len(ids) == 7 and len(set(ids.tolist())) == 7
        assert np.all(np.diff(d) >= 0)
        np.testing.assert_allclose(d, 1.0 - base[ids] @ q, atol=1e-5)


def test_graph_invariants():
    base, _, idx = _build()
    n = base.shape[0]
    assert len(idx._neighbors) == n
    for node, layers in enumerate(idx._neighbors):
        for layer, nl in enumerate(layers):
            cap = idx.M0 if layer == 0 else idx.M
            assert len(nl) <= cap
            assert node not in nl
            assert len(set(nl)) == len(nl)
            for m in nl:
                assert len(idx._neighbors[m]) > layer  # neighbour exists on this layer
    # entry point is on the top layer
    assert len(idx._neighbors[idx._entry]) - 1 == idx._max_level
    # layer 0 is connected from the entry point
    seen, todo = {idx._entry}, deque([idx._entry])
    while todo:
        c = todo.popleft()
        for m in idx._neighbors[c][0]:
            if m not in seen:
                seen.add(m); todo.append(m)
    assert len(seen) == n


def test_deterministic_with_seed():
    _, queries, a = _build(seed=4)
    _, _, b = _build(seed=4)
    assert a._neighbors == b._neighbors
    for q in queries[:5]:
        assert a.search(q, 5)[0].tolist() == b.search(q, 5)[0].tolist()
