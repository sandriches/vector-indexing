"""Download Simple English Wikipedia, chunk, embed and cache.

    uv run scripts/embed_corpus.py --articles 25000 --words 100
"""

import argparse

from vector_index.data import load_or_embed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=25000)
    ap.add_argument("--words", type=int, default=100)
    args = ap.parse_args()
    chunks, emb = load_or_embed(args.articles, args.words)
    print(f"{len(chunks)} chunks from {args.articles} articles -> embeddings {emb.shape} {emb.dtype}")
    print(f"example: [{chunks[0].title}] {chunks[0].text[:120]}...")


if __name__ == "__main__":
    main()
