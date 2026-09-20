import numpy as np

from vector_index.bench import ground_truth, recall_at_k, split_queries
from vector_index.brute import BruteForceIndex


def test_recall_at_k():
    truth = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k(truth, truth) == 1.0
    assert recall_at_k(np.array([[1, 2, 9], [7, 8, 9]]), truth) == 2 / 6
    # order within top-k must not matter
    assert recall_at_k(np.array([[3, 2, 1], [6, 5, 4]]), truth) == 1.0


def test_split_is_disjoint_and_complete():
    v = np.arange(50, dtype=np.float32).reshape(25, 2)
    base, q = split_queries(v, 5)
    assert base.shape == (20, 2) and q.shape == (5, 2)
    all_rows = {tuple(r) for r in np.vstack([base, q])}
    assert all_rows == {tuple(r) for r in v}


def test_ground_truth_recall_is_one_for_brute():
    rng = np.random.default_rng(3)
    v = rng.standard_normal((200, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    base, q = split_queries(v, 10)
    truth = ground_truth(base, q, 5)
    idx = BruteForceIndex()
    idx.build(base)
    pred = np.stack([idx.search(x, 5)[0] for x in q])
    assert recall_at_k(pred, truth) == 1.0
