import numpy as np

from vector_index.brute import BruteForceIndex
from vector_index.ivf import IVFIndex


def _normed(rng, n, d):
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_full_probe_is_exact():
    rng = np.random.default_rng(0)
    base, queries = _normed(rng, 2000, 16), _normed(rng, 30, 16)
    bf = BruteForceIndex(); bf.build(base)
    ivf = IVFIndex(nlist=16, nprobe=16); ivf.build(base)
    for q in queries:
        a, da = bf.search(q, 10)
        b, db = ivf.search(q, 10)
        assert a.tolist() == b.tolist()
        np.testing.assert_allclose(da, db, atol=1e-5)


def test_partial_probe_scans_fewer_and_stays_sorted():
    rng = np.random.default_rng(1)
    base, queries = _normed(rng, 2000, 16), _normed(rng, 10, 16)
    ivf = IVFIndex(nlist=32, nprobe=4); ivf.build(base)
    for q in queries:
        ids, d = ivf.search(q, 5)
        assert len(ids) == 5 and len(set(ids.tolist())) == 5
        assert np.all(np.diff(d) >= 0)
        assert ivf.last_scanned < base.shape[0]
        # returned distances are true distances
        np.testing.assert_allclose(d, 1.0 - base[ids] @ q, atol=1e-5)


def test_all_ids_stored_once():
    rng = np.random.default_rng(2)
    base = _normed(rng, 500, 8)
    ivf = IVFIndex(nlist=10, nprobe=1); ivf.build(base)
    assert sorted(ivf._ids.tolist()) == list(range(500))
    assert ivf._offsets[-1] == 500


def test_train_subsample():
    rng = np.random.default_rng(3)
    base = _normed(rng, 1000, 8)
    ivf = IVFIndex(nlist=8, nprobe=8, train_size=200); ivf.build(base)
    ids, _ = ivf.search(base[0], 1)
    assert ids[0] == 0
