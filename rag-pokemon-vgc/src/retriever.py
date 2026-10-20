"""
Retriever: búsqueda semántica sobre ChromaDB con filtros y umbral.

Aplica el prefijo "query: " de E5 a las consultas y devuelve
los chunks más similares con su score de similitud.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os
# Fix para redes con SSL interception (antivirus, proxy, universidad)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K,
    SIMILARITY_THRESHOLD,
    E5_QUERY_PREFIX,
)
from src.ingestion import get_embeddings


# ============================================================
# ESTRUCTURA DE RESULTADO
# ============================================================
@dataclass
class RetrievedChunk:
    """Un chunk recuperado con su score de similitud."""
    content: str
    score: float               # similitud coseno en [0, 1] (mayor = mejor)
    metadata: dict[str, Any]

    @property
    def source(self) -> str:
        return self.metadata.get("source", "desconocido")

    @property
    def breadcrumb(self) -> str:
        return self.metadata.get("breadcrumb", "")

    @property
    def page(self) -> int:
        return self.metadata.get("page", 0)

    @property
    def tag(self) -> str:
        return self.metadata.get("tag", "")

    @property
    def chunk_type(self) -> str:
        return self.metadata.get("chunk_type", "")


# ============================================================
# RETRIEVER
# ============================================================
class Retriever:
    """Encapsula la carga de ChromaDB y la búsqueda semántica."""

    def __init__(self) -> None:
        self.embeddings = get_embeddings()
        self.vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    def search(
        self,
        query: str,
        k: int = TOP_K,
        tag: str | None = None,
        threshold: float = SIMILARITY_THRESHOLD,
        chunk_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Busca chunks similares a la consulta.

        Args:
            query: pregunta del usuario.
            k: número máximo de chunks a recuperar.
            tag: filtrar por tag (ej: "vgc_rules"). None = buscar en todos.
            threshold: similitud mínima (0-1) para aceptar un chunk.
            chunk_types: filtrar por tipo ("text", "table", "table_row").

        Returns:
            Lista de RetrievedChunk ordenados por similitud descendente.
        """
        # E5 requiere prefijo "query:" al consultar
        query_with_prefix = f"{E5_QUERY_PREFIX}{query}"

        # Construir filtro de ChromaDB
        where_filter = self._build_where_filter(tag, chunk_types)

        # Búsqueda con scores de similitud
        # similarity_search_with_relevance_scores devuelve score en [0, 1]
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query_with_prefix,
            k=k,
            filter=where_filter,
        )

        retrieved: list[RetrievedChunk] = []
        for doc, score in results:
            if score < threshold:
                continue
            retrieved.append(RetrievedChunk(
                content=doc.page_content,
                score=float(score),
                metadata=doc.metadata,
            ))

        return retrieved

    @staticmethod
    def _build_where_filter(
        tag: str | None,
        chunk_types: list[str] | None,
    ) -> dict[str, Any] | None:
        """Construye el filtro 'where' de ChromaDB a partir de los parámetros."""
        conditions: list[dict[str, Any]] = []

        if tag is not None:
            conditions.append({"tag": tag})

        if chunk_types:
            if len(chunk_types) == 1:
                conditions.append({"chunk_type": chunk_types[0]})
            else:
                conditions.append({"chunk_type": {"$in": chunk_types}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def list_tags(self) -> list[str]:
        """Lista los tags disponibles en la colección."""
        data = self.vectorstore.get(include=["metadatas"])
        tags = {m.get("tag") for m in data["metadatas"] if m}
        return sorted(t for t in tags if t)

    def count(self) -> int:
        """Cuenta total de chunks en la colección."""
        return self.vectorstore._collection.count()


# ============================================================
# HELPERS DE CONVENIENCIA
# ============================================================
def format_retrieved_for_context(chunks: list[RetrievedChunk]) -> str:
    """
    Formatea los chunks recuperados como contexto para el LLM.
    Cada chunk lleva un header con su fuente para trazabilidad.
    """
    if not chunks:
        return "(No se encontró contexto relevante en la base de conocimiento)"

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[Fuente {i}] {chunk.source} | "
            f"Sección: {chunk.breadcrumb or 'N/A'} | "
            f"Página: {chunk.page} | "
            f"Tipo: {chunk.chunk_type} | "
            f"Similitud: {chunk.score:.3f}"
        )
        parts.append(f"{header}\n{chunk.content}")

    return "\n\n---\n\n".join(parts)


# ============================================================
# TEST RÁPIDO
# ============================================================
if __name__ == "__main__":
    retriever = Retriever()
    print(f"✅ Colección cargada: {retriever.count()} chunks")
    print(f"   Tags disponibles: {retriever.list_tags()}")

    test_queries = [
        "¿Cuántos Pokémon puedo llevar en un equipo VGC?",
        "¿Qué sanción recibe un jugador por slow play?",
        "¿Qué está prohibido en cuanto a vestimenta en los torneos?",
    ]

    for q in test_queries:
        print(f"\n❓ {q}")
        results = retriever.search(q, k=3)
        if not results:
            print("   ⚠️  Sin resultados por encima del umbral")
        for r in results:
            print(f"   [{r.score:.3f}] ({r.tag}) {r.breadcrumb[:60]}")
            print(f"      {r.content[:100].replace(chr(10), ' ')}...")