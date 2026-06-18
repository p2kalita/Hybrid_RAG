import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

MD_DIR = Path(__file__).parent / "data" / "markdown"
OUT_PATH = Path(__file__).parent / "indexes" / "corpus.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 300       # characters per chunk
CHUNK_OVERLAP = 100     # characters of overlap between consecutive chunks

# Ordered from largest structural break to smallest. "\n#" catches markdown
# headers of any level; pymupdf4llm also emits "---" for horizontal rules
# and tables use "|" rows, which naturally fall through to paragraph/line
# splitting since we don't want to break a table row mid-cell.
MARKDOWN_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=MARKDOWN_SEPARATORS,
    )


def load_markdown_files() -> list[Path]:
    files = sorted(MD_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(
            f"No .md files found in {MD_DIR}. Run 0_pdf_to_markdown.py first "
            f"(or drop .md files there directly)."
        )
    return files


def build_corpus() -> list[dict]:
    splitter = get_splitter()
    records = []

    for md_path in load_markdown_files():
        text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue
            records.append(
                {
                    "_id": f"{md_path.stem}::chunk{i}",
                    "source": md_path.name,
                    "text": chunk,
                }
            )
    return records


if __name__ == "__main__":
    records = build_corpus()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Chunked {len(records)} chunks from {MD_DIR} -> {OUT_PATH}")