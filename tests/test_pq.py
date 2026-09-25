import numpy as np

from vector_index.brute import BruteForceIndex
from vector_index.pq import IVFPQIndex, PQIndex, ProductQuantizer


def _lowrank(rng, n, d=32, rank=4):
    # structured data so quantization has something to exploit
    v = (rng.standard_normal((n, rank)) @ rng.standard_normal((rank, d))).astype(np.float32)
    v += 0.05 * rng.standard_normal(v.shape).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_codes_shape_and_roundtrip_error():
    rng = np.random.default_rng(0)
    X = _lowrank(rng, 3000)
    baseline = ((X - X.mean(0)) ** 2).sum(1).mean()
    prev = np.inf
    for m in (2, 4, 8):
        pq = ProductQuantizer(m=m, ksub=16, n_iter=10)
        pq.train(X)
        codes = pq.encode(X)
        assert codes.shape == (3000, m) and codes.dtype == np.uint8
        err = ((pq.decode(codes) - X) ** 2).sum(1).mean()
        assert err < baseline and err < prev
        prev = err


def test_adc_equals_ip_with_reconstruction():
    rng = np.random.default_rng(1)
    X = _lowrank(rng, 1000)
    pq = ProductQuantizer(m=4, ksub=16, n_iter=10)
    pq.train(X)
    codes = pq.encode(X)
    q = X[0]
    T = pq.ip_tables(q)
    adc = T[np.arange(4)[None, :], codes].sum(1)
    np.testing.assert_allclose(adc, pq.decode(codes) @ q, atol=1e-4)


def test_m_must_divide_d():
    pq = ProductQuantizer(m=5)
    try:
        pq.train(np.zeros((10, 32), dtype=np.float32))
        assert False
    except ValueError:
        pass


def test_pq_index_full_rerank_is_exact():
    rng = np.random.default_rng(2)
    base, queries = _lowrank(rng, 1500), _lowrank(rng, 10)
    bf = BruteForceIndex(); bf.build(base)
    idx = PQIndex(m=8, rerank=1500); idx.build(base)
    for q in queries:
        a, da = bf.search(q, 5)
        b, db = idx.search(q, 5)
        assert a.tolist() == b.tolist()
        np.testing.assert_allclose(da, db, atol=1e-5)


def test_pq_index_no_rerank_reasonable():
    rng = np.random.default_rng(3)
    base, queries = _lowrank(rng, 1500), _lowrank(rng, 20)
    bf = BruteForceIndex(); bf.build(base)
    # neighbours in this data are packed tightly (10th vs 11th differ by ~1e-3
    # in cosine distance) so exact top-10 recall without rerank is not
    # expected; what must hold is that finer subspaces do better.
    recall = {}
    for m in (4, 16):
        idx = PQIndex(m=m, rerank=0); idx.build(base)
        hits = 0
        for q in queries:
            ids, d = idx.search(q, 10)
            assert len(set(ids.tolist())) == 10 and np.all(np.diff(d) >= 0)
            hits += len(set(ids.tolist()) & set(bf.search(q, 10)[0].tolist()))
        recall[m] = hits / 200
        assert idx._codes_T.nbytes == base.shape[0] * m
    assert recall[16] > recall[4] > 0.3


def test_ivfpq_full_probe_full_rerank_is_exact():
    rng = np.random.default_rng(4)
    base, queries = _lowrank(rng, 2000), _lowrank(rng, 10)
    bf = BruteForceIndex(); bf.build(base)
    idx = IVFPQIndex(nlist=8, m=8, nprobe=8, rerank=2000, train_size=2000); idx.build(base)
    for q in queries:
        a, _ = bf.search(q, 5)
        b, _ = idx.search(q, 5)
        assert a.tolist() == b.tolist()


def test_ivfpq_structure_and_partial_probe():
    rng = np.random.default_rng(5)
    base, queries = _lowrank(rng, 2000), _lowrank(rng, 10)
    idx = IVFPQIndex(nlist=16, m=8, nprobe=4, rerank=0, train_size=2000); idx.build(base)
    assert sorted(idx._ids.tolist()) == list(range(2000))
    assert idx._offsets[-1] == 2000 and idx._codes.shape == (2000, 8)
    for q in queries:
        ids, d = idx.search(q, 5)
        assert len(set(ids.tolist())) == 5 and np.all(np.diff(d) >= 0)
        assert idx.last_scanned < 2000
