import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

INDEX_DIR = Path(__file__).parent / "indexes"
CORPUS_PATH = INDEX_DIR / "corpus.jsonl"
BM25_PATH = INDEX_DIR / "bm25.pkl"

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"{CORPUS_PATH} not found. Run 1_ingest.py first.")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_index(corpus: list[dict]) -> BM25Okapi:
    tokenized_docs = [tokenize(doc["text"]) for doc in corpus]
    return BM25Okapi(tokenized_docs)


def search_bm25(bm25: BM25Okapi, doc_ids: list[str], query: str, k: int = 10) -> list[tuple[str, float]]:
    """Return the top-k (doc_id, score) pairs for a query."""
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(doc_ids[i], float(scores[i])) for i in ranked]


if __name__ == "__main__":
    corpus = load_corpus()
    doc_ids = [doc["_id"] for doc in corpus]

    print(f"Building BM25 index over {len(corpus)} chunks...")
    bm25 = build_index(corpus)

    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids}, f)
    print(f"Saved BM25 index -> {BM25_PATH}")

    # quick smoke test
    query = corpus[0]["text"][:40]
    print(f"\nSmoke-test query: {query!r}\n")
    for i, (doc_id, score) in enumerate(search_bm25(bm25, doc_ids, query, k=3), 1):
        text = next(d["text"] for d in corpus if d["_id"] == doc_id)
        print(f"{i}. [{score:.3f}] {doc_id}\n   {text[:120]}...\n")