# Vector Index: learning project plan

A single Python repo that grows through four phases. Every phase plugs into
the same benchmark harness and is measured against brute-force search on the
same data, so each algorithm gets a concrete recall and latency number.

Language: Python 3.10+, NumPy for all vector maths. No vector-database
libraries (FAISS, hnswlib, etc.) in the core; they may be used only as an
optional reference to check our own results against.

## Goals

- Understand what a vector index is and why brute force stops being enough.
- Build IVF, HNSW and product quantization by hand and see the
  recall-versus-speed-versus-memory tradeoffs directly.
- Keep one harness the whole way through so results are comparable.

## Repo layout (target)

```
vector-index/
  PLAN.md
  pyproject.toml            # uv-managed
  src/vector_index/
    data.py                 # load / embed / cache datasets
    brute.py                # Phase 1: exact search
    ivf.py                  # Phase 2
    hnsw.py                 # Phase 3
    pq.py                   # Phase 4
    bench.py                # harness: recall@k, latency, build time, memory
  scripts/
    embed_corpus.py
    run_bench.py
  tests/
  data/                     # git-ignored: raw corpus, cached embeddings
  results/                  # benchmark outputs (CSV / markdown tables)
```

## Common interface

Every index implements the same protocol so the harness can treat them alike:

```python
class Index(Protocol):
    def build(self, vectors: np.ndarray) -> None: ...
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (ids, distances), both shape (k,)."""
```

## Dataset

Simple English Wikipedia (`wikimedia/wikipedia`, config `20231101.simple`):
one 157 MB parquet file, 241,787 articles. We take the first 25,000 articles
and split each into 100-word chunks, which gives roughly 100k chunks. Chunks
are embedded locally with `sentence-transformers/all-MiniLM-L6-v2` (384
dims, runs on the Apple GPU via MPS, no API key) and L2-normalised so cosine
similarity is a dot product.

Chunk counts at 100 words per chunk, for reference:

| articles | chunks |
|---|---|
| 5,000 | 36k |
| 10,000 | 54k |
| 25,000 | ~100k |
| 40,000 | 131k |

1,000 chunks are held out as queries so the trivial self-match never occurs.
Ground truth for recall is computed once by brute force and cached in
`data/`. From Phase 2 onwards we can optionally add a standard
ANN-benchmarks dataset (e.g. `glove-25-angular` or `sift-128-euclidean`)
which ships with precomputed ground truth, for comparison with published
numbers.

## Metrics (fixed from Phase 1)

- **recall@k**: fraction of true top-k neighbours returned (k = 10 default).
- **query latency**: p50 and p95 over a fixed set of 1,000 queries.
- **queries per second**, single-threaded.
- **build time** and **index memory** (bytes, estimated from array sizes).

Results are written to `results/<index>_<dataset>.csv` and a markdown table
is regenerated so phases can be compared at a glance.

---

## Phase 1: brute force + benchmark harness  (done)

The baseline. Not an index yet, but everything later is measured against it.

Deliverables:
1. `uv` project with `numpy`, `sentence-transformers`, `pytest`.
2. `data.py`: load corpus, chunk text, embed, L2-normalise, cache to `.npy`.
3. `brute.py`: exact cosine search via one matrix multiply and `argpartition`.
4. `bench.py`: computes the metrics above for any `Index`; caches ground truth.
5. A small CLI (`scripts/run_bench.py`) and a `results/` table with the
   brute-force row filled in.
6. Tests: brute-force top-k matches a naive per-row loop on a tiny array.

Definition of done: one command embeds the corpus, runs the harness on
brute force, and prints a table with recall@10 = 1.0 plus latency numbers.

Concepts to learn: cosine vs dot product vs L2 and why normalisation lets
you swap them; why `argpartition` beats a full sort; memory bandwidth as the
real cost of brute force.

## Phase 2: IVF (inverted file index)  (done)

Cluster the vectors with k-means, store each vector under its nearest
centroid, and at query time only scan the `nprobe` closest clusters.

Deliverables:
- Own k-means implementation in NumPy (Lloyd's algorithm, k-means++ init).
- `ivf.py` with `nlist` and `nprobe` parameters.
- Benchmark sweep over `nprobe` producing a recall-vs-QPS curve.

Definition of done: at some `nprobe`, IVF reaches recall@10 >= 0.9 while
being clearly faster than brute force on the 100k corpus.

Concepts: coarse quantization, the recall/speed knob, why more clusters
means cheaper scans but worse boundaries.

## Phase 3: HNSW (hierarchical navigable small world graph)  (done)

A layered proximity graph. Search greedily walks from an entry point in the
top layer down to layer 0, then does a beam search with width `ef`.

Deliverables:
- `hnsw.py` implementing insert (with `M`, `ef_construction`) and search
  (with `ef_search`). Pure Python plus NumPy for distances; speed is not the
  goal at first, correctness is.
- Recall-vs-QPS curve over `ef_search`, plotted against IVF and brute force.
- Optional: a flat (single-layer) NSW first, then add the hierarchy, to see
  what the layers buy you.

Definition of done: recall@10 >= 0.95 with a large speedup over brute force
in query count terms (distance computations per query), even if wall-clock
is limited by Python overhead.

Concepts: navigable small-world graphs, greedy search and local minima,
neighbour selection heuristics, why build is expensive and search is cheap.

## Phase 4: product quantization  (next)

Compress vectors by splitting each into `m` subvectors and replacing each
subvector with the index of its nearest codebook entry. Search uses
asymmetric distance computation with precomputed lookup tables.

Deliverables:
- `pq.py`: train codebooks (reuse Phase 2 k-means), encode, and search.
- Combine with IVF (IVF-PQ) so both the scan cost and memory shrink.
- Memory column in the results table becomes meaningful: compare bytes per
  vector for float32, PQ, and IVF-PQ.

Definition of done: a memory reduction of at least 8x versus float32 with a
measured, understood recall cost, and an optional re-ranking step using the
original vectors to recover recall.

Concepts: quantization error, asymmetric distance, lookup-table search, the
memory/recall tradeoff and re-ranking.

---

## Later ideas (not scheduled)

- Persist an index to disk and reload it.
- CLIP-embedded image search over a photo folder using the same indexes.
- Filtered search (metadata predicates) and how it interacts with each index.
- A minimal HTTP API in front of the best index.

## Progress log

- 2026-09-20: plan written.
- 2026-09-20: Phase 1 scaffolded: uv project, `data.py`, `brute.py`, `bench.py`,
  scripts, 9 tests passing. Verified end to end on a 500-article slice
  (3,569 chunks, recall@10 = 1.0). Full 25k-article embedding run started.
- 2026-09-20: Phase 1 done. 25k articles -> 98,579 chunks, embedded in 5 min on
  MPS. Brute force on 97,579 base vectors (1,000 held-out queries): recall@10
  1.0, p50 3.2 ms, ~314 QPS, 150 MB. This is the bar for Phases 2-4.
- 2026-09-20: Phase 2 done. `kmeans.py` (k-means++ init, Lloyd's, empty-cluster
  reseeding) and `ivf.py` (CSR-style cluster storage). Sweeps at nlist=1024 and
  256 in `results/`, curve in `results/recall_vs_qps.png`. nlist=1024 nprobe=16:
  recall@10 0.926 at 0.16 ms p50 (20x brute force). At an equal scan budget
  (~1/32 of the corpus) 1024 clusters beat 256 (0.959 vs 0.910): finer cells
  waste less of the probe on far-away vectors. Build is 11 s for 1024 clusters.
- 2026-09-23: Phase 3 done. `hnsw.py`: layered graph, greedy descent, beam
  search on layer 0, paper's diversity heuristic for link selection. Build on
  97,579 vectors: 242 s (M=16, efc=100), pickled to `data/` so ef sweeps skip
  the rebuild. Layer populations 91517 / 5671 / 359 / 30 / 2, mean layer-0
  degree 25.5. ef=50: recall@10 0.981 touching 902 vectors (~1%) per query,
  0.64 ms p50. ef=400 reaches 0.999. Harness now records mean `dist_comps`
  per query. In distance computations HNSW beats IVF clearly (recall 0.98 at
  ~900 comps vs IVF needing ~14k for 0.99); in wall clock IVF wins at the
  same recall because its scan is one matmul while HNSW pays Python overhead
  per hop. That gap is the cost of the interpreter, not the algorithm.
