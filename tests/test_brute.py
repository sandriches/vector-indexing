import numpy as np

from vector_index.brute import BruteForceIndex


def _normed(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_matches_naive_loop():
    rng = np.random.default_rng(0)
    base = _normed(rng, 500, 16)
    queries = _normed(rng, 20, 16)
    idx = BruteForceIndex()
    idx.build(base)
    for q in queries:
        ids, dists = idx.search(q, 7)
        naive = np.array([1.0 - float(q @ b) for b in base])
        expect = np.argsort(naive)[:7]
        assert ids.tolist() == expect.tolist()
        np.testing.assert_allclose(dists, naive[expect], rtol=1e-5)
        assert np.all(np.diff(dists) >= 0)


def test_batch_agrees_with_single():
    rng = np.random.default_rng(1)
    base = _normed(rng, 300, 8)
    queries = _normed(rng, 5, 8)
    idx = BruteForceIndex()
    idx.build(base)
    bids, _ = idx.search_batch(queries, 4)
    for q, row in zip(queries, bids):
        ids, _ = idx.search(q, 4)
        assert ids.tolist() == row.tolist()


def test_k_larger_than_base():
    rng = np.random.default_rng(2)
    base = _normed(rng, 3, 4)
    idx = BruteForceIndex()
    idx.build(base)
    ids, _ = idx.search(base[0], 10)
    assert len(ids) == 3 and ids[0] == 0
