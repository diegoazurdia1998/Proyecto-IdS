"""
Pipeline de ingesta de PDFs para el RAG.

Flujo:
    PDF → limpieza → detección de headers → árbol jerárquico
        → chunking de texto + tablas → embeddings E5 → ChromaDB
"""
from __future__ import annotations

import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Iterable

import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import (
    DATA_DIR,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    E5_PASSAGE_PREFIX,
    FONT_SIZE_TO_HEADING_LEVEL,
    FONT_SIZES_NOISE,
    PAGES_TO_SKIP,
    ENCODING_FIXES,
    PDF_CONFIGS,
    ChunkingConfig,
)


# ============================================================
# UTILIDADES DE LIMPIEZA
# ============================================================
def fix_encoding(text: str) -> str:
    """Corrige caracteres mal codificados en los PDFs de Pokémon."""
    for bad, good in ENCODING_FIXES.items():
        text = text.replace(bad, good)
    return text


def clean_text(text: str) -> str:
    """Normaliza espacios, saltos de línea y caracteres de control."""
    # Reemplazar múltiples saltos por doble salto (párrafos)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Colapsar espacios múltiples
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Eliminar espacios al final de línea
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def is_noise_span(text: str, size: float) -> bool:
    """Detecta números de página decorativos y otros ruidos."""
    stripped = text.strip()
    if not stripped:
        return True
    # Fuentes de ruido conocidas (18pt, 19pt)
    if round(size, 1) in FONT_SIZES_NOISE and stripped.isdigit():
        return True
    # Página con solo número
    if stripped.isdigit() and len(stripped) <= 3:
        return True
    return False


# ============================================================
# ESTRUCTURA DE BLOQUES
# ============================================================
@dataclass
class Block:
    """Unidad atómica extraída del PDF: párrafo, heading o tabla."""
    kind: str               # "heading" | "text" | "table"
    text: str
    page: int
    level: int | None = None       # Solo para headings (1, 2, 3)
    section_number: str | None = None  # Ej: "2.1.1"


@dataclass
class Section:
    """Nodo del árbol jerárquico del documento."""
    level: int
    number: str
    title: str
    page: int
    paragraphs: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)
    parent: "Section | None" = None

    @property
    def breadcrumb(self) -> str:
        """Ruta completa desde la raíz hasta esta sección."""
        parts = []
        node = self
        while node is not None and node.level > 0:
            parts.append(f"{node.number} {node.title}".strip())
            node = node.parent
        return " > ".join(reversed(parts))


# ============================================================
# EXTRACCIÓN DE BLOQUES POR PDF (PyMuPDF)
# ============================================================
def extract_blocks_from_pdf(pdf_path: Path) -> list[Block]:
    """Extrae bloques de texto y headings clasificados por tamaño de fuente."""
    doc = fitz.open(pdf_path)
    blocks: list[Block] = []

    for page_num, page in enumerate(doc, start=1):
        if page_num <= PAGES_TO_SKIP:
            continue  # Saltar portada e índice

        page_dict = page.get_text("dict")

        for block in page_dict["blocks"]:
            if block.get("type") != 0:  # no es texto
                continue

            # Reunir el texto de todas las líneas/spans del bloque
            block_lines: list[str] = []
            block_font_sizes: list[float] = []

            for line in block["lines"]:
                line_text_parts: list[str] = []
                for span in line["spans"]:
                    text = span["text"]
                    size = round(span["size"], 1)
                    if is_noise_span(text, size):
                        continue
                    line_text_parts.append(text)
                    block_font_sizes.append(size)

                line_text = "".join(line_text_parts).strip()
                if line_text:
                    block_lines.append(line_text)

            if not block_lines:
                continue

            block_text = fix_encoding(" ".join(block_lines))
            block_text = clean_text(block_text)

            if not block_text:
                continue

            # Determinar si es heading según fuente dominante
            if block_font_sizes:
                dominant_size = max(set(block_font_sizes),
                                    key=block_font_sizes.count)
                level = FONT_SIZE_TO_HEADING_LEVEL.get(dominant_size)
            else:
                level = None

            if level is not None and len(block_text) < 120:
                # Es un heading: intentar extraer número de sección
                number, title = parse_heading(block_text)
                blocks.append(Block(
                    kind="heading",
                    text=title,
                    page=page_num,
                    level=level,
                    section_number=number,
                ))
            else:
                blocks.append(Block(
                    kind="text",
                    text=block_text,
                    page=page_num,
                ))

    doc.close()
    return blocks


def parse_heading(text: str) -> tuple[str, str]:
    """
    Separa '2.1.1 Cumplimiento de...' en ('2.1.1', 'Cumplimiento de...').
    Si no hay número, retorna ('', text).
    """
    match = re.match(r"^(\d+(?:\.\d+)*)\s+(.*)$", text.strip())
    if match:
        return match.group(1), match.group(2).strip()
    return "", text.strip()


# ============================================================
# EXTRACCIÓN DE TABLAS POR PDF (pdfplumber)
# ============================================================
def extract_tables_from_pdf(pdf_path: Path) -> list[Block]:
    """Extrae tablas como bloques, convertidas a Markdown."""
    blocks: list[Block] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num <= PAGES_TO_SKIP:
                continue

            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue

                md = table_to_markdown(table)
                if md.strip():
                    blocks.append(Block(
                        kind="table",
                        text=md,
                        page=page_num,
                    ))

    return blocks


def table_to_markdown(table: list[list[str | None]]) -> str:
    """Convierte una tabla extraída por pdfplumber a Markdown."""
    cleaned_rows: list[list[str]] = []
    for row in table:
        cleaned = [
            (cell or "").replace("\n", " ").strip()
            for cell in row
        ]
        # Saltar filas completamente vacías
        if any(cell for cell in cleaned):
            cleaned_rows.append(cleaned)

    if len(cleaned_rows) < 2:
        return ""

    # Normalizar ancho de filas
    max_cols = max(len(r) for r in cleaned_rows)
    normalized = [r + [""] * (max_cols - len(r)) for r in cleaned_rows]

    header = normalized[0]
    separator = ["---"] * max_cols
    body = normalized[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return fix_encoding("\n".join(lines))


# ============================================================
# CONSTRUCCIÓN DEL ÁRBOL JERÁRQUICO
# ============================================================
def build_section_tree(blocks: list[Block]) -> Section:
    """Convierte una lista plana de bloques en un árbol de secciones."""
    root = Section(level=0, number="", title="__root__", page=0)
    stack: list[Section] = [root]
    counters: dict[int, int] = {}  # para numerar headings sin número explícito

    for block in blocks:
        if block.kind == "heading" and block.level is not None:
            number = block.section_number or ""
            # Nivel del heading
            level = block.level

            # Pop del stack hasta encontrar un padre con menor nivel
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()

            parent = stack[-1]
            new_section = Section(
                level=level,
                number=number,
                title=block.text,
                page=block.page,
                parent=parent,
            )
            parent.children.append(new_section)
            stack.append(new_section)
        else:
            # Contenido: se asigna a la sección actual (top del stack)
            current = stack[-1]
            if block.kind == "table":
                current.tables.append(block.text)
            else:
                current.paragraphs.append(block.text)

    return root


def iter_sections(root: Section) -> Iterable[Section]:
    """Recorre el árbol en profundidad (DFS)."""
    for child in root.children:
        yield child
        yield from iter_sections(child)


# ============================================================
# CHUNKING DE TEXTO
# ============================================================
def build_text_chunks(
    sections: Iterable[Section],
    cfg: ChunkingConfig,
    source_name: str,
) -> list[Document]:
    """Genera chunks de texto a partir de las secciones."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size * 4,      # aproximación: 1 token ≈ 4 chars
        chunk_overlap=cfg.chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )

    documents: list[Document] = []

    for section in sections:
        if not section.paragraphs:
            continue

        breadcrumb = section.breadcrumb
        body = "\n\n".join(section.paragraphs)

        # Prepend del breadcrumb para enriquecer el contexto del embedding
        enriched = f"[{breadcrumb}]\n{body}"

        chunks = splitter.split_text(enriched)

        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": source_name,
                    "tag": cfg.tag,
                    "page": section.page,
                    "chunk_type": "text",
                    "section_number": section.number,
                    "section_l1": _section_at_level(section, 1),
                    "section_l2": _section_at_level(section, 2),
                    "section_l3": _section_at_level(section, 3),
                    "breadcrumb": breadcrumb,
                    "chunk_index": i,
                },
            ))

    return documents


def _section_at_level(section: Section, level: int) -> str:
    """Devuelve el título de la sección ancestra en un nivel dado."""
    node = section
    while node is not None:
        if node.level == level:
            return f"{node.number} {node.title}".strip()
        node = node.parent
    return ""


# ============================================================
# CHUNKING DE TABLAS
# ============================================================
def build_table_chunks(
    sections: Iterable[Section],
    cfg: ChunkingConfig,
    source_name: str,
) -> list[Document]:
    """Genera chunks de tablas. Si table_row_as_chunk, una fila = un chunk."""
    documents: list[Document] = []

    for section in sections:
        breadcrumb = section.breadcrumb

        for table_md in section.tables:
            if cfg.table_row_as_chunk:
                rows = _split_markdown_table(table_md)
                header = rows[0] if rows else ""
                for i, row in enumerate(rows[1:], start=1):
                    content = f"[{breadcrumb}]\n{header}\n{row}"
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": source_name,
                            "tag": cfg.tag,
                            "page": section.page,
                            "chunk_type": "table_row",
                            "section_number": section.number,
                            "breadcrumb": breadcrumb,
                            "row_index": i,
                        },
                    ))
            else:
                content = f"[{breadcrumb}]\n{table_md}"
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": source_name,
                        "tag": cfg.tag,
                        "page": section.page,
                        "chunk_type": "table",
                        "section_number": section.number,
                        "breadcrumb": breadcrumb,
                    },
                ))

    return documents


def _split_markdown_table(table_md: str) -> list[str]:
    """Divide una tabla Markdown en sus líneas (header + separador + filas)."""
    return [line for line in table_md.split("\n") if line.strip()]


# ============================================================
# PIPELINE COMPLETO POR PDF
# ============================================================
def process_pdf(pdf_path: Path, cfg: ChunkingConfig) -> list[Document]:
    """Procesa un PDF completo y devuelve los chunks listos para ChromaDB."""
    print(f"\n📄 Procesando: {pdf_path.name}  [tag={cfg.tag}]")

    # 1) Extraer bloques de texto
    text_blocks = extract_blocks_from_pdf(pdf_path)
    print(f"   Bloques de texto: {len(text_blocks)}")

    # 2) Extraer tablas
    table_blocks = extract_tables_from_pdf(pdf_path) if cfg.extract_tables else []
    print(f"   Tablas detectadas: {len(table_blocks)}")

    # 3) Construir árbol jerárquico (texto + tablas)
    all_blocks = sorted(text_blocks + table_blocks, key=lambda b: (b.page,))
    # Nota: ordenar por página es suficiente; dentro de página se respeta el orden
    tree = build_section_tree(all_blocks)
    sections = list(iter_sections(tree))
    print(f"   Secciones construidas: {len(sections)}")

    # 4) Chunking
    text_chunks = build_text_chunks(sections, cfg, pdf_path.name)
    table_chunks = build_table_chunks(sections, cfg, pdf_path.name)
    print(f"   Chunks de texto: {len(text_chunks)}")
    print(f"   Chunks de tabla: {len(table_chunks)}")

    return text_chunks + table_chunks


# ============================================================
# EMBEDDINGS + CHROMADB
# ============================================================
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Crea el modelo de embeddings E5.
    Nota: los prefijos 'passage:' / 'query:' se aplican manualmente al
    indexar y consultar (ver apply_e5_prefix).
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def apply_e5_prefix(text: str, is_query: bool = False) -> str:
    """Añade el prefijo correcto de E5 (obligatorio para buena calidad)."""
    from src.config import E5_PASSAGE_PREFIX, E5_QUERY_PREFIX
    prefix = E5_QUERY_PREFIX if is_query else E5_PASSAGE_PREFIX
    return f"{prefix}{text}"


def ingest_to_chromadb(
    documents: list[Document],
    reset: bool = True,
) -> Chroma:
    """Carga los chunks a ChromaDB con embeddings E5."""
    print(f"\n🔢 Generando embeddings para {len(documents)} chunks...")
    print("   (la primera vez descarga el modelo, ~500 MB)")

    embeddings = get_embeddings()

    # Aplicar prefijo E5 a los textos de los documentos
    for doc in documents:
        doc.page_content = apply_e5_prefix(doc.page_content, is_query=False)

    # Crear/reemplazar la colección
    if reset and CHROMA_DIR.exists():
        print(f"   Borrando colección previa en {CHROMA_DIR}")
        import shutil
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"✅ Cargados {len(documents)} chunks en ChromaDB")
    print(f"   Colección: {CHROMA_COLLECTION_NAME}")
    print(f"   Directorio: {CHROMA_DIR}")
    return vectorstore


# ============================================================
# ENTRYPOINT
# ============================================================
def run_ingestion(reset: bool = True) -> None:
    """Ejecuta la ingesta completa de los 4 PDFs."""
    all_documents: list[Document] = []

    for pdf_name, cfg in PDF_CONFIGS.items():
        pdf_path = DATA_DIR / pdf_name
        if not pdf_path.exists():
            print(f"⚠️  No encontrado: {pdf_path}")
            continue

        chunks = process_pdf(pdf_path, cfg)
        all_documents.extend(chunks)

    if not all_documents:
        print("❌ No se generaron documentos. Abortando.")
        return

    print(f"\n📊 Total de chunks: {len(all_documents)}")
    print("   Distribución por tag:")
    from collections import Counter
    tag_counter = Counter(d.metadata["tag"] for d in all_documents)
    for tag, count in tag_counter.most_common():
        print(f"     - {tag}: {count}")

    ingest_to_chromadb(all_documents, reset=reset)

    # Guardar un resumen
    summary_path = CHROMA_DIR.parent / "preprocess" / "ingestion_summary.json"
    summary_path.parent.mkdir(exist_ok=True, parents=True)
    summary = {
        "total_chunks": len(all_documents),
        "by_tag": dict(tag_counter),
        "by_source": dict(Counter(d.metadata["source"] for d in all_documents)),
        "by_type": dict(Counter(d.metadata["chunk_type"] for d in all_documents)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"📝 Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    run_ingestion(reset=True)