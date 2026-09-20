"""Load Simple English Wikipedia, chunk it, embed it, cache everything.

The dump is a single ~157 MB parquet file on HuggingFace. We take the first
N articles, split each into fixed-size word chunks, and embed the chunks
with a small local sentence-transformers model. Embeddings are L2-normalised
so cosine similarity is just a dot product.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HF_REPO = "wikimedia/wikipedia"
HF_FILE = "20231101.simple/train-00000-of-00001.parquet"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Chunk:
    article_id: int
    title: str
    text: str


def download_dump(cache_dir: Path = DATA_DIR) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            HF_REPO, HF_FILE, repo_type="dataset", cache_dir=cache_dir / "hf"
        )
    )


def chunk_text(text: str, words_per_chunk: int, min_words: int = 20) -> list[str]:
    """Split on whitespace into runs of `words_per_chunk` words.

    A trailing fragment shorter than `min_words` is dropped rather than kept
    as a near-empty chunk.
    """
    words = re.split(r"\s+", text.strip())
    words = [w for w in words if w]
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        piece = words[i : i + words_per_chunk]
        if len(piece) >= min_words:
            chunks.append(" ".join(piece))
    return chunks


def build_corpus(n_articles: int, words_per_chunk: int, cache_dir: Path = DATA_DIR) -> list[Chunk]:
    cache = cache_dir / f"corpus_a{n_articles}_w{words_per_chunk}.jsonl"
    if cache.exists():
        with cache.open() as f:
            return [Chunk(**json.loads(line)) for line in f]

    import pyarrow.parquet as pq

    table = pq.read_table(download_dump(cache_dir), columns=["id", "title", "text"])
    chunks: list[Chunk] = []
    for row in table.slice(0, n_articles).to_pylist():
        for piece in chunk_text(row["text"], words_per_chunk):
            chunks.append(Chunk(int(row["id"]), row["title"], piece))

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__) + "\n")
    return chunks


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    print(f"embedding {len(texts)} texts with {model_name} on {model.device}")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return np.ascontiguousarray(emb, dtype=np.float32)


def embedding_cache_path(n_articles: int, words_per_chunk: int, model_name: str, cache_dir: Path = DATA_DIR) -> Path:
    tag = model_name.split("/")[-1]
    return cache_dir / f"emb_a{n_articles}_w{words_per_chunk}_{tag}.npy"


def load_or_embed(
    n_articles: int,
    words_per_chunk: int = 100,
    model_name: str = DEFAULT_MODEL,
    cache_dir: Path = DATA_DIR,
) -> tuple[list[Chunk], np.ndarray]:
    """Return (chunks, embeddings), building and caching both if needed."""
    chunks = build_corpus(n_articles, words_per_chunk, cache_dir)
    path = embedding_cache_path(n_articles, words_per_chunk, model_name, cache_dir)
    if path.exists():
        emb = np.load(path)
        if emb.shape[0] == len(chunks):
            return chunks, emb
    emb = embed_texts([c.text for c in chunks], model_name)
    np.save(path, emb)
    return chunks, emb
