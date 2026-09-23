"""Benchmark harness: recall@k, latency, throughput, build time, memory.

Ground truth comes from brute force and is cached per (dataset, k) so later
phases don't pay for it again. Queries are held out from the corpus, so the
trivial "nearest neighbour is yourself" case never occurs.
"""

import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .brute import BruteForceIndex
from .index import Index

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


@dataclass
class Dataset:
    name: str
    base: np.ndarray      # vectors to index
    queries: np.ndarray   # held-out vectors
    truth: np.ndarray     # (n_queries, k) ids into base, closest first


def split_queries(vectors: np.ndarray, n_queries: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Hold out `n_queries` random rows as queries; the rest form the base."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(vectors.shape[0])
    q_idx, b_idx = perm[:n_queries], perm[n_queries:]
    return vectors[b_idx], vectors[q_idx]


def ground_truth(base: np.ndarray, queries: np.ndarray, k: int, batch: int = 512) -> np.ndarray:
    bf = BruteForceIndex()
    bf.build(base)
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for i in range(0, queries.shape[0], batch):
        ids, _ = bf.search_batch(queries[i : i + batch], k)
        out[i : i + batch] = ids
    return out


def make_dataset(name: str, vectors: np.ndarray, n_queries: int, k: int, cache_dir: Path) -> Dataset:
    base, queries = split_queries(vectors, n_queries)
    cache = cache_dir / f"truth_{name}_q{n_queries}_k{k}.npy"
    if cache.exists():
        truth = np.load(cache)
    else:
        truth = ground_truth(base, queries, k)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, truth)
    return Dataset(name, base, queries, truth)


def recall_at_k(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean over queries of |pred ∩ truth| / k."""
    k = truth.shape[1]
    hits = 0
    for p, t in zip(pred, truth):
        hits += len(set(p[:k].tolist()) & set(t.tolist()))
    return hits / (truth.shape[0] * k)


@dataclass
class BenchResult:
    index: str
    dataset: str
    n_base: int
    dim: int
    k: int
    recall: float
    p50_ms: float
    p95_ms: float
    qps: float
    build_s: float
    memory_mb: float
    dist_comps: float = 0.0   # mean vectors compared per query (0 if index doesn't report)
    params: str = ""

    def row(self) -> dict:
        return asdict(self)


def benchmark(index: Index, ds: Dataset, k: int, params: str = "", build: bool = True) -> BenchResult:
    """Build (unless `build=False`, for sweeping a query-time knob) and time
    `search` over every held-out query one at a time."""
    build_s = 0.0
    if build:
        t0 = time.perf_counter()
        index.build(ds.base)
        build_s = time.perf_counter() - t0

    # warm up
    index.search(ds.queries[0], k)

    pred = np.empty((ds.queries.shape[0], k), dtype=np.int64)
    lat = np.empty(ds.queries.shape[0])
    comps = np.zeros(ds.queries.shape[0])
    for i, q in enumerate(ds.queries):
        t = time.perf_counter()
        ids, _ = index.search(q, k)
        lat[i] = time.perf_counter() - t
        pred[i, : len(ids)] = ids
        comps[i] = getattr(index, "last_dist_comps", 0)

    return BenchResult(
        index=index.name,
        dataset=ds.name,
        n_base=ds.base.shape[0],
        dim=ds.base.shape[1],
        k=k,
        recall=recall_at_k(pred, ds.truth),
        p50_ms=float(np.percentile(lat, 50) * 1e3),
        p95_ms=float(np.percentile(lat, 95) * 1e3),
        qps=float(1.0 / lat.mean()),
        build_s=build_s,
        memory_mb=index.memory_bytes() / 1e6,
        dist_comps=float(comps.mean()),
        params=params,
    )


def append_result(res: BenchResult, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{res.index}_{res.dataset}.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res.row().keys()))
        if new:
            w.writeheader()
        w.writerow(res.row())
    return path


def render_table(results_dir: Path = RESULTS_DIR) -> str:
    rows = []
    for path in sorted(results_dir.glob("*.csv")):
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        return "(no results yet)"
    cols = ["index", "params", "dataset", "n_base", "k", "recall", "p50_ms", "p95_ms", "qps", "dist_comps", "build_s", "memory_mb"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            try:
                fv = float(v)
                v = f"{fv:.3f}" if c == "recall" else f"{fv:.0f}" if c == "dist_comps" else f"{fv:.2f}" if "." in v else v
            except ValueError:
                pass
            cells.append(v)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_table(results_dir: Path = RESULTS_DIR) -> Path:
    path = results_dir / "RESULTS.md"
    path.write_text("# Benchmark results\n\n" + render_table(results_dir) + "\n")
    return path
