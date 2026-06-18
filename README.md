# Hybrid RAG with Groq + HuggingFace Embeddings

A from-scratch hybrid retrieval pipeline for customer support CSVs, adapted from
[daveebbelaar/ai-cookbook's hybrid-retrieval example](https://github.com/daveebbelaar/ai-cookbook/tree/main/knowledge/hybrid-retrieval).

**What changed from the original:**
| | Original (ai-cookbook) | v1.0 | v2.0 |
|---|---|---|---|
| Input format | FiQA finance dataset (parquet) | `.pdf` / `.md` files | Customer support tickets (`.csv`) |
| Ingestion | — | `1_pdf_2_markdown.py` | `1_ingest_csv.py` |
| Chunking method | No chunking used | `RecursiveCharacterTextSplitter` (300 chars, 100 overlap) | Sentence-window chunking with metadata prefix (5 sentences, 1 overlap) |
| Dense embeddings | OpenAI `text-embedding-3-small` (API) | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free, CPU) | Same |
| Sparse retrieval | BM25 (`rank_bm25`) | Same | Same |
| Fusion | Reciprocal Rank Fusion (RRF) | Same | Same |
| Generation | None (retrieval-only demo) | Added: Groq-hosted LLM generates the final answer | Same |

## How it works

```
data/*.csv
      │
      ▼
1_ingest_csv.py  → chunks tickets → indexes/corpus.jsonl
      │
      ├──► 3_bm25.py   → keyword index    → indexes/bm25.pkl
      │
      └──► 4_embed.py  → semantic index   → indexes/embeddings.npy
                  │
                  ▼
        5_hybrid.py → fuses BM25 + dense rankings with RRF
                  │
                  ▼
        rag.py → retrieves top chunks → sends to Groq → returns grounded answer
```

BM25 catches exact terms, codes, and names. Dense embeddings catch paraphrases and synonyms. RRF combines their *rankings* (not raw scores, which live on incompatible scales) so you get both kinds of matches in one result set. Groq then generates a final answer constrained to that retrieved context.

## Chunking strategy

Each CSV row is a structured support ticket with four fields: `Body`, `Department`, `Priority`, and `Tags`. The ingester splits the body at sentence boundaries using NLTK's Punkt tokenizer, groups sentences into overlapping windows (5 sentences, 1-sentence overlap), and prefixes every chunk with a single-line metadata header:

```
Department: Technical Support | Priority: high | Tags: Account, Outage

Dear Customer Support Team, I am writing to report...
```

This means BM25 can keyword-match on department names, priority levels, and tags in every chunk, while the embedding model encodes metadata and body together for richer semantic similarity. Rows with empty bodies are skipped with a warning.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your free Groq key from https://console.groq.com/keys
```

## Usage

1. **Add your CSV files** — drop `.csv` files into `data/`. Each file must have `Body`, `Department`, `Priority`, and `Tags` columns.

2. **Build the indexes** (run once, or whenever you change `data/`):
   ```bash
   python 1_ingest_csv.py
   python 3_bm25.py
   python 4_embed.py
   ```

3. **Try retrieval alone** (no API key needed):
   ```bash
   python 5_hybrid.py
   ```

4. **Ask questions with full RAG** (needs `GROQ_API_KEY` in `.env`):
   ```bash
   python rag.py "What are the most common billing issues?"
   # or run with no args for an interactive prompt loop
   python rag.py
   ```

## Notes & things you'll likely want to tune

- **Chunking** (`1_ingest_csv.py`): sentence-window with `MAX_SENTENCES=5` and `OVERLAP_SENTENCES=1`. Increase `MAX_SENTENCES` if ticket bodies are long and coherent; decrease it for dense, multi-topic tickets where tighter retrieval precision matters.
- **Input folder**: `DATA_DIR` in `1_ingest_csv.py` defaults to `./data/`. Change the constant at the top of the file if your CSVs live elsewhere.
- **Embedding model** (`4_embed.py`): swap `MODEL_NAME` for any sentence-transformers model on HuggingFace, e.g. `BAAI/bge-base-en-v1.5` for higher quality at more compute cost. No code changes needed elsewhere — dimensions are handled automatically.
- **Groq model** (`rag.py`): `GROQ_MODEL` defaults to `llama-3.3-70b-versatile`. Check your Groq console for the current list of available models if you want something faster/cheaper or larger.
- **RRF candidate pool / k** (`5_hybrid.py`): `candidate_pool=50` controls how many results each retriever contributes before fusion; `RRF_K=60` is the standard constant from the original RRF paper. Both are safe defaults but tunable.
- **Re-embedding**: `4_embed.py` caches embeddings and skips recomputation if the corpus's doc IDs haven't changed. Delete `indexes/embeddings.npy` and `indexes/doc_ids.json` to force a rebuild, or re-run `1_ingest_csv.py` after changing `data/` (new/changed doc IDs trigger automatic re-embedding).