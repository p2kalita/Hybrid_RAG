import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, runs on CPU

INDEX_DIR = Path(__file__).parent / "indexes"
CORPUS_PATH = INDEX_DIR / "corpus.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
DOC_IDS_PATH = INDEX_DIR / "doc_ids.json"

_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model once per process."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"{CORPUS_PATH} not found. Run 1_ingest.py first.")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Embed a list of texts and return an (N, dim) float32 array, L2-normalized
    so cosine similarity becomes a plain dot product."""
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def build_or_load_index(corpus: list[dict]) -> tuple[np.ndarray, list[str]]:
    doc_ids = [doc["_id"] for doc in corpus]

    if EMBEDDINGS_PATH.exists() and DOC_IDS_PATH.exists():
        cached_ids = json.loads(DOC_IDS_PATH.read_text())
        if cached_ids == doc_ids:
            print(f"Loading cached embeddings from {EMBEDDINGS_PATH}")
            return np.load(EMBEDDINGS_PATH), doc_ids
        print("Corpus changed since last embedding run — re-embedding.")

    texts = [doc["text"] for doc in corpus]
    print(f"Embedding {len(texts)} chunks with {MODEL_NAME} (local, CPU)...")
    doc_embeddings = embed_texts(texts, show_progress=True)

    np.save(EMBEDDINGS_PATH, doc_embeddings)
    DOC_IDS_PATH.write_text(json.dumps(doc_ids))
    print(f"Saved embeddings -> {EMBEDDINGS_PATH}")
    return doc_embeddings, doc_ids


def search_dense(
    query: str, doc_embeddings: np.ndarray, doc_ids: list[str], k: int = 10
) -> list[tuple[str, float]]:
    """Return the top-k (doc_id, similarity) pairs for a query."""
    query_vec = embed_texts([query])[0]
    scores = doc_embeddings @ query_vec  # already normalized -> dot product = cosine sim
    top_k = np.argsort(-scores)[:k]
    return [(doc_ids[i], float(scores[i])) for i in top_k]


if __name__ == "__main__":
    corpus = load_corpus()
    doc_embeddings, doc_ids = build_or_load_index(corpus)

    query = corpus[0]["text"][:40]
    print(f"\nSmoke-test query: {query!r}\n")
    for i, (doc_id, score) in enumerate(search_dense(query, doc_embeddings, doc_ids, k=3), 1):
        text = next(d["text"] for d in corpus if d["_id"] == doc_id)
        print(f"{i}. [{score:.3f}] {doc_id}\n   {text[:120]}...\n")