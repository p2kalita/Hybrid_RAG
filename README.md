# Hybrid RAG with Groq + HuggingFace Embeddings

A from-scratch hybrid retrieval pipeline, adapted from
[daveebbelaar/ai-cookbook's hybrid-retrieval example](https://github.com/daveebbelaar/ai-cookbook/tree/main/knowledge/hybrid-retrieval).

**What changed from the original:**
| | Original (ai-cookbook) | This version |
|---|---|---|
| Dense embeddings | OpenAI `text-embedding-3-small` (API) | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free, CPU) |
| Sparse retrieval | BM25 (`rank_bm25`) | Same |
| Fusion | Reciprocal Rank Fusion (RRF) | Same |
| Generation | None (retrieval-only demo) | Added: Groq-hosted LLM generates the final answer |
| Corpus | FiQA finance dataset (parquet) | Your own `.pdf` files in `data/pdf/` |

## How it works

```
data/pdf/*.pdf, *.md
      │
      ▼
1_pdf_2_markdown.py  
      |
      |
      ├──► 2_chunk.py  → chunks your docs → indexes/corpus.jsonl
      │
      ├──► 3_bm25.py   → keyword index → indexes/bm25.pkl
      │
      └──► 4_embed.py  → semantic index (local model) → indexes/embeddings.npy
                  │
                  ▼
        5_hybrid.py → fuses BM25 + dense rankings with RRF
                  │
                  ▼
        rag.py → retrieves top chunks → sends to Groq → returns grounded answer
```

BM25 catches exact terms, codes, and names. Dense embeddings catch paraphrases and synonyms. RRF combines their *rankings* (not raw scores, which live on incompatible scales) so you get both kinds of matches in one result set. Groq then generates a final answer constrained to that retrieved context.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your free Groq key from https://console.groq.com/keys
```

## Usage

1. **Add your documents** — drop `.pdf` or `.md` files into `data/pdf` (one sample file is included so you can test immediately).

2. **Build the indexes** (run once, or whenever you change `data/`):
   ```bash
   python 1_pdf_2_markdown.py
   python 2_chunk.py
   python 3_bm25.py
   python 4_embed.py
   ```

3. **Try retrieval alone** (no API key needed):
   ```bash
   python 5_hybrid.py
   ```

4. **Ask questions with full RAG** (needs `GROQ_API_KEY` in `.env`):
   ```bash
   python rag.py "What is this paper about?"
   # or run with no args for an interactive prompt loop
   python rag.py
   ```

## Notes & things you'll likely want to tune

- **Chunking** (`2_chunk.py`): character-based with paragraph/sentence-aware breaking, 300 chars with 100 overlap. Adjust `CHUNK_SIZE`/`CHUNK_OVERLAP` for your documents — long-form prose may want larger chunks, dense reference docs smaller ones.
- **Embedding model** (`4_embed.py`): swap `MODEL_NAME` for any sentence-transformers model on HuggingFace, e.g. `sentence-transformers/all-MiniLM-L6-v2` for somewhat better quality at similar speed, or `BAAI/bge-base-en-v1.5` for higher quality at more compute cost. No code changes needed elsewhere — dimensions are handled automatically.
- **Groq model** (`rag.py`): `GROQ_MODEL` defaults to `llama-3.3-70b-versatile`. Check your Groq console for the current list of available models if you want something faster/cheaper or larger.
- **RRF candidate pool / k** (`5_hybrid.py`): `candidate_pool=50` controls how many results each retriever contributes before fusion; `RRF_K=60` is the standard constant from the original RRF paper. Both are safe defaults but tunable.
- **Re-embedding**: `4_embed.py` caches embeddings and skips recomputation if the corpus's doc IDs haven't changed. Delete `indexes/embeddings.npy` and `indexes/doc_ids.json` to force a rebuild, or just re-run `1_ingest.py` after changing `data/` (new/changed doc IDs trigger automatic re-embedding).
