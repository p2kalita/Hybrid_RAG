import json
import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from importlib import import_module

embed_mod = import_module("4_embed")
hybrid_mod = import_module("5_hybrid")

load_dotenv()

INDEX_DIR = Path(__file__).parent / "indexes"
CORPUS_PATH = INDEX_DIR / "corpus.jsonl"
BM25_PATH = INDEX_DIR / "bm25.pkl"

GROQ_MODEL = "llama-3.3-70b-versatile"  # solid default; swap for any model your Groq account supports
TOP_K = 5

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- Base your answer strictly on the context below. Do not use outside knowledge.
- If the context doesn't contain enough information to answer, say so plainly.
- Cite which source each piece of your answer comes from, e.g. (source: filename.txt).
- Be concise and direct.
"""


def load_corpus() -> dict[str, dict]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"{CORPUS_PATH} not found.")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return {(d := json.loads(line))["_id"]: d for line in f}


def load_bm25():
    if not BM25_PATH.exists():
        raise FileNotFoundError(f"{BM25_PATH} not found.")
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["doc_ids"]


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context block for the prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {c['source']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(
    query: str,
    corpus_by_id: dict,
    bm25,
    bm25_doc_ids: list[str],
    doc_embeddings,
    dense_doc_ids: list[str],
    groq_client: Groq,
    k: int = TOP_K,
) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks via hybrid search, then ask Groq to answer using them."""
    hits = hybrid_mod.search_hybrid(
        query, bm25, bm25_doc_ids, doc_embeddings, dense_doc_ids, k=k
    )
    retrieved = [corpus_by_id[doc_id] for doc_id, _score in hits]
    context = build_context(retrieved)

    user_prompt = f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    return answer, retrieved


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(
            "GROQ_API_KEY not set."
        )
        sys.exit(1)

    groq_client = Groq(api_key=api_key)

    print("Loading indexes...")
    corpus_by_id = load_corpus()
    bm25, bm25_doc_ids = load_bm25()
    corpus_list = list(corpus_by_id.values())
    doc_embeddings, dense_doc_ids = embed_mod.build_or_load_index(corpus_list)
    print(f"Ready. {len(corpus_by_id)} chunks indexed.\n")

    query_arg = " ".join(sys.argv[1:]).strip()

    def run_query(query: str):
        answer, retrieved = answer_question(
            query, corpus_by_id, bm25, bm25_doc_ids, doc_embeddings, dense_doc_ids, groq_client
        )
        print(f"\nAnswer:\n{answer}\n")
        print("Retrieved chunks:")
        for c in retrieved:
            print(f"  - {c['_id']} ({c['source']})")
        print()

    if query_arg:
        run_query(query_arg)
        return

    print("Interactive mode. Type a question, or 'exit' to quit.\n")
    while True:
        query = input("> ").strip()
        if query.lower() in {"exit", "quit", ""}:
            break
        run_query(query)


if __name__ == "__main__":
    main()