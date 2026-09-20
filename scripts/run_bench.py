"""Run the harness on one index and append to results/.

    uv run scripts/run_bench.py --index brute --articles 25000
"""

import argparse

from vector_index.bench import append_result, benchmark, make_dataset, write_table
from vector_index.brute import BruteForceIndex
from vector_index.data import DATA_DIR, load_or_embed

INDEXES = {
    "brute": lambda: BruteForceIndex(),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", choices=INDEXES, default="brute")
    ap.add_argument("--articles", type=int, default=25000)
    ap.add_argument("--words", type=int, default=100)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    _, emb = load_or_embed(args.articles, args.words)
    ds = make_dataset(f"simplewiki_a{args.articles}_w{args.words}", emb, args.queries, args.k, DATA_DIR)
    res = benchmark(INDEXES[args.index](), ds, args.k)
    append_result(res)
    print(write_table().read_text())


if __name__ == "__main__":
    main()
