# vector-index

Vector search indexes written from scratch in NumPy, benchmarked against
brute force on Simple English Wikipedia. Started as a way to learn how this
stuff actually works. Plan and progress in [PLAN.md](PLAN.md).

## Usage

```bash
uv sync --group dev
uv run pytest

# download + embed the corpus (~5 min first time, cached in data/)
uv run scripts/embed_corpus.py

# benchmark an index, writes results/RESULTS.md
uv run scripts/run_bench.py --index brute

# poke at it
uv run scripts/search.py
```
