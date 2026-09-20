"""Interactive sanity check: type a query, see the nearest chunks.

    uv run scripts/search.py --articles 25000
"""

import argparse

from vector_index.brute import BruteForceIndex
from vector_index.data import embed_texts, load_or_embed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=25000)
    ap.add_argument("--words", type=int, default=100)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    chunks, emb = load_or_embed(args.articles, args.words)
    idx = BruteForceIndex()
    idx.build(emb)
    while True:
        try:
            q = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        qv = embed_texts([q], show_progress=False)[0]
        ids, dists = idx.search(qv, args.k)
        for i, d in zip(ids, dists):
            c = chunks[i]
            print(f"  {1 - d:.3f}  [{c.title}] {c.text[:140]}...")


if __name__ == "__main__":
    main()
