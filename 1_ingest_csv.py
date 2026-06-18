"""
Chunking strategy: metadata-prefixed sentence-window chunking for customer support CSVs
========================================================================================
 
Problem
-------
Each CSV row is a structured support ticket with four fields: Body, Department,
Priority, and Tags. Naive text splitting (fixed character or token windows) fails
here for two reasons:
 
  1. It can split mid-sentence, breaking grammatical context that both BM25 and
     embedding models depend on for accurate scoring.
  2. It discards the structured metadata (department, priority, tags), which are
     high-signal retrieval features — a query like "high priority billing issues"
     should match on those fields, not just the free-text body.
 
Approach
--------
Each ticket body is split at sentence boundaries using NLTK's Punkt tokenizer,
then grouped into overlapping windows of MAX_SENTENCES sentences. Every resulting
chunk is prefixed with a single-line header that inlines the structured metadata:
 
    Department: Billing and Payments | Priority: high | Tags: Billing, Payment
 
This means:
  - BM25 can keyword-match on department names, priority levels, and tags across
    every chunk without separate field indexing.
  - The dense embedder encodes metadata and body together, so semantic similarity
    is computed over the full context of a ticket, not the body in isolation.
  - The LLM receives self-contained chunks: each one identifies what kind of ticket
    it came from, making generated answers easier to attribute.
 
Sliding window with overlap
---------------------------
Window size (MAX_SENTENCES = 5) and overlap (OVERLAP_SENTENCES = 1) are chosen for
typical support ticket bodies of 3–15 sentences:
 
  - Most tickets fit in a single chunk (≤ 5 sentences), so no splitting occurs.
  - For longer tickets, a 1-sentence overlap ensures a sentence that straddles a
    window boundary is represented in both adjacent chunks, preventing context loss
    at retrieval boundaries.
  - Increasing MAX_SENTENCES reduces chunk count but risks exceeding the embedding
    model's effective context window for very long bodies; decreasing it increases
    retrieval precision at the cost of more chunks per ticket.
 
Chunk ID scheme
---------------
IDs are deterministic: MD5(source_filename::row_index)[:12] + "_c{chunk_index}".
Re-ingesting the same CSV produces identical IDs, which allows 4_embed.py to detect
unchanged corpora and skip re-embedding via its cached_ids == doc_ids check.
 
Output schema (corpus.jsonl)
-----------------------------
Each line is a JSON object with:
  _id         — deterministic chunk identifier
  source      — originating CSV filename
  row_index   — original row number in the CSV (for traceability)
  text        — header + body chunk (the field indexed by BM25 and the embedder)
  department  — raw department value (for downstream metadata filtering)
  priority    — raw priority value (for downstream metadata filtering)
  tags        — list of tag strings (for downstream metadata filtering)
"""

import ast
import hashlib
import json
from pathlib import Path

import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

INDEX_DIR = Path(__file__).parent / "indexes"
INDEX_DIR.mkdir(exist_ok=True)

CORPUS_PATH = INDEX_DIR / "corpus.jsonl"

MAX_SENTENCES = 5
OVERLAP_SENTENCES = 1


def parse_tags(raw) -> list[str]:
    if isinstance(raw, list):
        return raw
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass
    return [
        x.strip().strip("'\"")
        for x in str(raw).strip("[]").split(",")
        if x.strip()
    ]


def make_chunk_id(source: str, row_idx: int, chunk_idx: int) -> str:
    base = hashlib.md5(f"{source}::{row_idx}".encode()).hexdigest()[:12]
    return f"{base}_c{chunk_idx}"


def build_header(row: pd.Series) -> tuple[str, dict]:
    tags = parse_tags(row.get("Tags", []))
    department = str(row.get("Department", "Unknown")).strip()
    priority = str(row.get("Priority", "Unknown")).strip()

    metadata = {
        "department": department,
        "priority": priority,
        "tags": tags,
    }

    header = f"Department: {department} | Priority: {priority} | Tags: {', '.join(tags)}"
    return header, metadata


def semantic_chunk(
    body: str,
    max_sentences: int = MAX_SENTENCES,
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> list[str]:
    body = str(body).strip()
    if not body:
        return []

    sentences = sent_tokenize(body)
    if len(sentences) <= max_sentences:
        return [body]

    chunks = []
    start = 0
    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunks.append(" ".join(sentences[start:end]))
        if end == len(sentences):
            break
        start = end - overlap_sentences

    return chunks


def ingest_csv(csv_path: str) -> list[dict]:
    source = Path(csv_path).name
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    records = []

    for row_idx, row in df.iterrows():
        header, metadata = build_header(row)
        body = str(row.get("Body", "")).strip()
        body_chunks = semantic_chunk(body)

        if not body_chunks:
            print(f"  [warn] row {row_idx} has empty body — skipping")
            continue

        for chunk_idx, chunk_body in enumerate(body_chunks):
            text = header + "\n\n" + chunk_body
            records.append({
                "_id": make_chunk_id(source, row_idx, chunk_idx),
                "source": source,
                "row_index": int(row_idx),
                "text": text,
                "department": metadata["department"],
                "priority": metadata["priority"],
                "tags": metadata["tags"],
            })

    return records


def main():

    CSV_DIR = Path(__file__).parent / "data"

    if not CSV_DIR.exists():
        print(f"Folder not found: {CSV_DIR}")
        return

    csv_files = list(CSV_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {CSV_DIR}")
        return

    all_records = []

    for csv_path in csv_files:
        print(f"Ingesting {csv_path.name}")
        records = ingest_csv(str(csv_path))
        all_records.extend(records)
        print(f"  -> {len(records)} chunks")

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_records)} chunks to {CORPUS_PATH}")


if __name__ == "__main__":
    main()