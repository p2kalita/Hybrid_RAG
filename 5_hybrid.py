"""
Reciprocal Rank Fusion (RRF). Combine BM25 and dense retrieval into one
ranked list.

The naive idea is 'average the scores', but BM25 scores are unbounded and
cosine similarities sit in [0, 1]. The fix is to fuse RANKINGS, not scores.

    rrf_score(d) = sum over each retriever r of 1 / (k + rank_r(d))

k is a smoothing constant, conventionally 60. The original 2009 paper called
it 'simple but effective' and that is still the consensus in 2026.

More info: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
"""


import json
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer

from importlib import import_module

bm25_mod = import_module("3_bm25")
embed_mod = import_module("4_embed")

INDEX_DIR = Path(__file__).parent / "indexes"
CORPUS_PATH = INDEX_DIR / "corpus.jsonl"
BM25_PATH = INDEX_DIR / "bm25.pkl"

RRF_K = 60


def load_corpus() -> list[dict]:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_bm25():
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["doc_ids"]


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of doc_ids into one score per doc_id."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def search_hybrid(
    query: str,
    bm25,
    bm25_doc_ids: list[str],
    doc_embeddings,
    dense_doc_ids: list[str],
    k: int = 10,
    candidate_pool: int = 50,
) -> list[tuple[str, float]]:
    """
    Retrieve top candidates from both retrievers, fuse with RRF, return top-k.

    candidate_pool controls how many results we pull from *each* retriever
    before fusing — wider than k so RRF has enough signal to work with.
    """
    bm25_hits = bm25_mod.search_bm25(bm25, bm25_doc_ids, query, k=candidate_pool)
    dense_hits = embed_mod.search_dense(query, doc_embeddings, dense_doc_ids, k=candidate_pool)

    bm25_ranked = [doc_id for doc_id, _ in bm25_hits]
    dense_ranked = [doc_id for doc_id, _ in dense_hits]

    fused = rrf_fuse([bm25_ranked, dense_ranked])
    return fused[:k]


if __name__ == "__main__":
    corpus = load_corpus()
    corpus_by_id = {d["_id"]: d for d in corpus}

    bm25, bm25_doc_ids = load_bm25()
    doc_embeddings, dense_doc_ids = embed_mod.build_or_load_index(corpus)

    query = "What is this document about?"
    print(f"\nQuery: {query}\n")
    results = search_hybrid(query, bm25, bm25_doc_ids, doc_embeddings, dense_doc_ids, k=5)
    for i, (doc_id, score) in enumerate(results, 1):
        text = corpus_by_id[doc_id]["text"]
        print(f"{i}. [rrf={score:.4f}] {doc_id}\n   {text[:150]}...\n")