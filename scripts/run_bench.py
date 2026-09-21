"""Run the harness on one index and append to results/.

    uv run scripts/run_bench.py --index brute --articles 25000
"""

import argparse

from vector_index.bench import append_result, benchmark, make_dataset, write_table
from vector_index.brute import BruteForceIndex
from vector_index.data import DATA_DIR, load_or_embed
from vector_index.ivf import IVFIndex

INDEXES = ["brute", "ivf"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", choices=INDEXES, default="brute")
    ap.add_argument("--nlist", type=int, default=1024, help="ivf: number of clusters")
    ap.add_argument("--nprobe", default="8", help="ivf: clusters to scan; comma list sweeps without rebuilding")
    ap.add_argument("--articles", type=int, default=25000)
    ap.add_argument("--words", type=int, default=100)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    _, emb = load_or_embed(args.articles, args.words)
    ds = make_dataset(f"simplewiki_a{args.articles}_w{args.words}", emb, args.queries, args.k, DATA_DIR)
    if args.index == "brute":
        append_result(benchmark(BruteForceIndex(), ds, args.k))
    elif args.index == "ivf":
        nprobes = [int(x) for x in args.nprobe.split(",")]
        ivf = IVFIndex(nlist=args.nlist, nprobe=nprobes[0])
        res = benchmark(ivf, ds, args.k, params=ivf.params())
        append_result(res)
        for nprobe in nprobes[1:]:
            ivf.nprobe = nprobe
            r = benchmark(ivf, ds, args.k, params=ivf.params(), build=False)
            r.build_s = res.build_s
            append_result(r)
    print(write_table().read_text())


if __name__ == "__main__":
    main()
