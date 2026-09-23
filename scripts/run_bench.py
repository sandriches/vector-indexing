"""Run the harness on one index and append to results/.

    uv run scripts/run_bench.py --index brute --articles 25000
"""

import argparse
import pickle

from vector_index.bench import append_result, benchmark, make_dataset, write_table
from vector_index.brute import BruteForceIndex
from vector_index.data import DATA_DIR, load_or_embed
from vector_index.hnsw import HNSWIndex
from vector_index.ivf import IVFIndex

INDEXES = ["brute", "ivf", "hnsw"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", choices=INDEXES, default="brute")
    ap.add_argument("--nlist", type=int, default=1024, help="ivf: number of clusters")
    ap.add_argument("--nprobe", default="8", help="ivf: clusters to scan; comma list sweeps without rebuilding")
    ap.add_argument("--M", type=int, default=16, help="hnsw: links per node")
    ap.add_argument("--efc", type=int, default=100, help="hnsw: ef_construction")
    ap.add_argument("--ef", default="50", help="hnsw: ef_search; comma list sweeps without rebuilding")
    ap.add_argument("--no-cache", action="store_true", help="hnsw: rebuild even if a pickled graph exists in data/")
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
    elif args.index == "hnsw":
        efs = [int(x) for x in args.ef.split(",")]
        cache = DATA_DIR / f"hnsw_{ds.name}_M{args.M}_efc{args.efc}.pkl"
        if cache.exists() and not args.no_cache:
            with cache.open("rb") as f:
                hnsw, build_s = pickle.load(f)
            print(f"loaded {cache.name} (built in {build_s:.0f}s)")
            hnsw.ef_search = efs[0]
            res = benchmark(hnsw, ds, args.k, params=hnsw.params(), build=False)
            res.build_s = build_s
        else:
            hnsw = HNSWIndex(M=args.M, ef_construction=args.efc, ef_search=efs[0])
            res = benchmark(hnsw, ds, args.k, params=hnsw.params())
            with cache.open("wb") as f:
                pickle.dump((hnsw, res.build_s), f)
        print(hnsw.stats())
        append_result(res)
        for ef in efs[1:]:
            hnsw.ef_search = ef
            r = benchmark(hnsw, ds, args.k, params=hnsw.params(), build=False)
            r.build_s = res.build_s
            append_result(r)
    print(write_table().read_text())


if __name__ == "__main__":
    main()
