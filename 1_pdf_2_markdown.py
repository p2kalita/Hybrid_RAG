from pathlib import Path

import pymupdf4llm

PDF_DIR = Path(__file__).parent / "data" / "pdfs"
MD_DIR = Path(__file__).parent / "data" / "markdown"

PDF_DIR.mkdir(parents=True, exist_ok=True)
MD_DIR.mkdir(parents=True, exist_ok=True)


def convert_pdf(pdf_path: Path) -> Path:
    """Convert a single PDF to a Markdown file and return the output path."""
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    out_path = MD_DIR / f"{pdf_path.stem}.md"
    out_path.write_text(md_text, encoding="utf-8")
    return out_path


def convert_all() -> list[Path]:
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_DIR}. Add your .pdf files there and re-run."
        )

    out_paths = []
    for pdf_path in pdf_files:
        print(f"Converting {pdf_path.name} ...")
        out_path = convert_pdf(pdf_path)
        out_paths.append(out_path)
        print(f"  -> {out_path}")
    return out_paths


if __name__ == "__main__":
    out_paths = convert_all()
    print(f"\nConverted {len(out_paths)} PDF(s) -> {MD_DIR}")