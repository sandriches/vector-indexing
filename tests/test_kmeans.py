import numpy as np

from vector_index.kmeans import assign, kmeans, kmeans_pp_init


def _blobs(rng, k=5, per=40, d=8, spread=0.05):
    centres = rng.standard_normal((k, d)) * 3
    X = np.concatenate([c + spread * rng.standard_normal((per, d)) for c in centres])
    y = np.repeat(np.arange(k), per)
    return X.astype(np.float32), y, centres.astype(np.float32)


def test_assign_matches_naive():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 6)).astype(np.float32)
    C = rng.standard_normal((7, 6)).astype(np.float32)
    labels, d = assign(X, C, chunk=16)
    for x, lab, dd in zip(X, labels, d):
        naive = ((C - x) ** 2).sum(axis=1)
        assert lab == naive.argmin()
        assert abs(dd - naive.min()) < 1e-3


def test_pp_init_picks_distinct_points():
    rng = np.random.default_rng(1)
    X, _, _ = _blobs(rng)
    C = kmeans_pp_init(X, 5, rng)
    assert C.shape == (5, 8)
    assert len({tuple(np.round(c, 4)) for c in C}) == 5


def test_kmeans_recovers_well_separated_blobs():
    rng = np.random.default_rng(2)
    X, y, centres = _blobs(rng)
    C, labels = kmeans(X, 5, n_iter=30, seed=2)
    # each true blob maps to exactly one found cluster
    for b in range(5):
        assert len(set(labels[y == b].tolist())) == 1
    assert len(set(labels.tolist())) == 5
    # found centroids sit on true centres
    for c in centres:
        assert np.min(np.linalg.norm(C - c, axis=1)) < 0.1


def test_kmeans_no_empty_clusters():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    C, labels = kmeans(X, 40, n_iter=10, seed=3)
    assert C.shape == (40, 4)
    assert np.bincount(labels, minlength=40).min() > 0
