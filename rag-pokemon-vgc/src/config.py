"""
Configuración central del RAG.
Todos los parámetros del pipeline se definen aquí.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()


# ============================================================
# PATHS DEL PROYECTO
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "chroma_db")


# ============================================================
# CONFIGURACIÓN DE EMBEDDINGS
# ============================================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
EMBEDDING_DIM = 768  # dimensión de multilingual-e5-base

# E5 requiere prefijos específicos (documentos vs consultas)
E5_PASSAGE_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "


# ============================================================
# CONFIGURACIÓN DEL LLM (Groq)
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))


# ============================================================
# CONFIGURACIÓN DE CHROMADB
# ============================================================
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pokemon_vgc_rules")


# ============================================================
# CONFIGURACIÓN DE RETRIEVAL
# ============================================================
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))


# ============================================================
# CONFIGURACIÓN DE CHUNKING POR PDF
# ============================================================
# Cada PDF tiene su propia configuración basada en el análisis estructural.
# Los tamaños están en TOKENS (no caracteres).

@dataclass
class ChunkingConfig:
    """Configuración de chunking para un PDF específico."""
    tag: str                          # Tag que se asigna a todos sus chunks
    chunk_size: int                   # Tamaño objetivo en tokens
    chunk_overlap: int                # Solapamiento en tokens
    extract_tables: bool = True       # Si extraer tablas como chunks aparte
    table_row_as_chunk: bool = False  # Si True, cada fila de tabla = 1 chunk


# Estrategia por PDF (basada en análisis con analyze_pdfs.py e inspect_structure.py)
PDF_CONFIGS: dict[str, ChunkingConfig] = {
    "play-pokemon-penalty-guidelines-es.pdf": ChunkingConfig(
        tag="penalty_guidelines",
        chunk_size=600,
        chunk_overlap=100,
        extract_tables=True,
        table_row_as_chunk=True,   # Matriz infracción→sanción: 1 fila = 1 chunk
    ),
    "play-pokemon-standards-of-conduct-es.pdf": ChunkingConfig(
        tag="standards_conduct",
        chunk_size=700,
        chunk_overlap=120,
        extract_tables=True,
        table_row_as_chunk=False,
    ),
    "play-pokemon-tournament-rules-handbook-es.pdf": ChunkingConfig(
        tag="general_rules",
        chunk_size=800,
        chunk_overlap=150,
        extract_tables=True,
        table_row_as_chunk=False,
    ),
    "play-pokemon-vgc-tournament-handbook-es.pdf": ChunkingConfig(
        tag="vgc_rules",
        chunk_size=700,
        chunk_overlap=130,
        extract_tables=True,
        table_row_as_chunk=False,
    ),
}


# ============================================================
# REGLAS DE PARSEO POR FUENTE
# ============================================================
# Basado en inspect_structure.py:
#   20pt = nivel 1 (##)
#   16pt = nivel 2 (###)
#   14pt = nivel 3 (####)
#   11pt = cuerpo de texto
FONT_SIZE_TO_HEADING_LEVEL = {
    20.0: 1,   # #
    16.0: 2,   # ##
    14.0: 3,   # ###
}

# Tamaños de fuente que son ruido (números de página decorativos)
FONT_SIZES_NOISE = {18.0, 19.0}

# Páginas a saltar al inicio (portada + índice)
PAGES_TO_SKIP = 3


# ============================================================
# FIX DE ENCODING (PDFs rotos: ż→¿, Ń→Ñ, etc.)
# ============================================================
ENCODING_FIXES = {
    "ż": "¿",
    "Ń": "Ñ",
    "ń": "ñ",
    "Ć": "Á",   # sospecha; ajustar si aparece
    "ć": "á",
}


# ============================================================
# VALIDACIÓN
# ============================================================
def validate_config() -> None:
    """Valida que la configuración esté completa antes de arrancar."""
    errors = []

    if not GROQ_API_KEY:
        errors.append(
            "❌ GROQ_API_KEY no está configurada. "
            "Cópiala en el archivo .env"
        )

    if not DATA_DIR.exists():
        errors.append(f"❌ DATA_DIR no existe: {DATA_DIR}")

    pdfs_found = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    if not pdfs_found:
        errors.append(f"❌ No hay PDFs en {DATA_DIR}")

    expected = set(PDF_CONFIGS.keys())
    found = {p.name for p in pdfs_found}
    missing = expected - found
    extra = found - expected

    if missing:
        errors.append(f"⚠️  PDFs esperados pero no encontrados: {missing}")
    if extra:
        errors.append(f"⚠️  PDFs encontrados sin config: {extra}")

    if errors:
        print("\n".join(errors))
        raise SystemExit(1)

    print("✅ Configuración válida")
    print(f"   - DATA_DIR: {DATA_DIR}")
    print(f"   - CHROMA_DIR: {CHROMA_DIR}")
    print(f"   - Embeddings: {EMBEDDING_MODEL}")
    print(f"   - LLM: {LLM_MODEL}")
    print(f"   - PDFs encontrados: {len(pdfs_found)}")


if __name__ == "__main__":
    validate_config()