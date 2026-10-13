"""
Diagnóstico de PDFs para diseñar la estrategia de chunking del RAG.
Uso: python scripts/analyze_pdfs.py
"""
import re
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import requests

import fitz  # PyMuPDF
import pdfplumber

DATA_DIR = Path("data")
REPORT_PATH = Path("preprocess/pdf_analysis_report.json")



def analyze_with_pymupdf(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    info = {
        "file": pdf_path.name,
        "pages": doc.page_count,
        "metadata": doc.metadata,
        "is_encrypted": doc.is_encrypted,
        "needs_ocr_suspected": False,
        "text_per_page": [],
        "tables_suspected": 0,
        "image_blocks": 0,
        "header_footer_candidates": [],
        "font_sizes": Counter(),
        "num_columns_suspected": 0,
    }

    for i, page in enumerate(doc):
        text = page.get_text("text")
        info["text_per_page"].append(len(text))
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] == 1:
                info["image_blocks"] += 1
            elif b["type"] == 0:
                for line in b["lines"]:
                    for span in line["spans"]:
                        info["font_sizes"][round(span["size"], 1)] += len(span["text"])

    avg_chars = sum(info["text_per_page"]) / max(len(info["text_per_page"]), 1)
    info["avg_chars_per_page"] = round(avg_chars, 1)
    info["needs_ocr_suspected"] = avg_chars < 200

    # Detectar headers/footers repetidos (primera y última línea de cada página)
    first_lines, last_lines = [], []
    for page in doc:
        lines = [l.strip() for l in page.get_text("text").split("\n") if l.strip()]
        if lines:
            first_lines.append(lines[0])
            last_lines.append(lines[-1])
    info["header_footer_candidates"] = {
        "top_repeated": [l for l, c in Counter(first_lines).most_common(3) if c > 2],
        "bottom_repeated": [l for l, c in Counter(last_lines).most_common(3) if c > 2],
    }

    # Top fuentes más usadas (probables títulos)
    info["font_sizes"] = dict(info["font_sizes"].most_common(8))
    doc.close()
    return info


def analyze_with_pdfplumber(pdf_path: Path) -> dict:
    result = {"tables_detected": 0, "pages_with_tables": []}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.find_tables()
            if tables:
                result["tables_detected"] += len(tables)
                result["pages_with_tables"].append(i + 1)
    return result


def detect_numbering_pattern(pdf_path: Path) -> dict:
    """Detecta si el documento está numerado por secciones (1., 1.1, A., etc.)."""
    doc = fitz.open(pdf_path)
    full = "\n".join(p.get_text("text") for p in doc)
    patterns = {
        "numbered_sections": len(re.findall(r"^\s*\d+\.\s+[A-ZÁÉÍÓÚÑ]", full, re.M)),
        "subsections": len(re.findall(r"^\s*\d+\.\d+\s+", full, re.M)),
        "articles": len(re.findall(r"^\s*(Artículo|Sección|Capítulo)\s+\d+", full, re.M | re.I)),
        "bullets": len(re.findall(r"^\s*[•\-\*]\s+", full, re.M)),
        "all_caps_headings": len(re.findall(r"^[A-ZÁÉÍÓÚÑ\s]{8,}$", full, re.M)),
    }
    doc.close()
    return patterns


def main():
    download_missing_pdfs()

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"❌ No hay PDFs en {DATA_DIR.resolve()}")
        return

    report = []
    for pdf in pdfs:
        print(f"\n📄 Analizando: {pdf.name}")
        entry = analyze_with_pymupdf(pdf)
        entry.update(analyze_with_pdfplumber(pdf))
        entry["structure"] = detect_numbering_pattern(pdf)
        report.append(entry)
        print(f"   Páginas: {entry['pages']} | "
              f"Chars/pág: {entry['avg_chars_per_page']} | "
              f"¿OCR?: {entry['needs_ocr_suspected']} | "
              f"Tablas: {entry['tables_detected']}")

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n✅ Reporte guardado en: {REPORT_PATH}")


if __name__ == "__main__":
    main()