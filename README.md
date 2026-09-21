# vector-indexing - an educational POC

Vector search indexes written from scratch in NumPy, benchmarked against
brute force on Simple English Wikipedia. Intended as a learning tool. Plan and progress in [PLAN.md](PLAN.md).

## Usage

```bash
uv sync --group dev
uv run pytest

# download + embed the corpus (~5 min first time, cached in data/)
uv run scripts/embed_corpus.py

# benchmark an index, writes results/RESULTS.md
uv run scripts/run_bench.py --index brute
uv run scripts/run_bench.py --index ivf --nlist 1024 --nprobe 1,4,16,64

# recall vs qps plot -> results/recall_vs_qps.png
uv run scripts/plot_results.py

# poke at it
uv run scripts/search.py
```
