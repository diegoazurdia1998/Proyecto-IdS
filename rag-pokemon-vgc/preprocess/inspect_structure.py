"""
scripts/inspect_structure.py
Inspecciona jerarquía, headers y muestra tablas crudas.
"""
import json
import re
from pathlib import Path
from collections import Counter

import fitz
import pdfplumber

DATA_DIR = Path("data")
OUT_DIR = Path("preprocess")
OUT_DIR.mkdir(exist_ok=True)


def dump_headings_and_fonts(pdf_path: Path, max_lines: int = 60):
    """Muestra las fuentes más grandes (probables títulos) y líneas candidatas a heading."""
    doc = fitz.open(pdf_path)
    font_counter = Counter()
    headings = []

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    size = round(span["size"], 1)
                    font_counter[size] += len(text)
                    # Candidato a heading: fuente grande + línea corta + no termina en punto
                    if size >= 13 and len(text) < 90 and not text.endswith("."):
                        headings.append({
                            "page": page_num,
                            "size": size,
                            "text": text,
                            "bold": "Bold" in span["font"] or "Black" in span["font"],
                        })
    doc.close()

    print(f"\n=== {pdf_path.name} ===")
    print("Top tamaños de fuente (tamaño: chars):")
    for size, count in font_counter.most_common(8):
        print(f"  {size:>5} pt : {count:>6} chars")

    print(f"\nPrimeros {max_lines} candidatos a heading:")
    for h in headings[:max_lines]:
        marker = "B" if h["bold"] else " "
        print(f"  p{h['page']:>3} | {h['size']:>5}pt | {marker} | {h['text'][:80]}")

    return {"fonts": dict(font_counter.most_common(10)), "headings_sample": headings[:40]}


def dump_table_samples(pdf_path: Path, max_tables: int = 3):
    """Muestra las primeras tablas detectadas para ver su forma."""
    print(f"\n--- Tablas de ejemplo en {pdf_path.name} ---")
    with pdfplumber.open(pdf_path) as pdf:
        count = 0
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                if count >= max_tables:
                    return
                print(f"\n  [Tabla pág {page_num} #{t_idx}] filas={len(table)}")
                for row in table[:6]:
                    cleaned = [(c or "").replace("\n", " ")[:35] for c in row]
                    print(f"    {cleaned}")
                count += 1


def main():
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    summary = {}
    for pdf in pdfs:
        summary[pdf.name] = dump_headings_and_fonts(pdf)
        dump_table_samples(pdf)

    (OUT_DIR / "structure_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"\n✅ Guardado: {OUT_DIR / 'structure_report.json'}")


if __name__ == "__main__":
    main()