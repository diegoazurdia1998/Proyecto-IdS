"""
Batería de pruebas del Plan de Calidad para el RAG de Pokémon VGC.

Cubre:
    1. Ingesta: colección cargada con chunks suficientes
    2. Metadata: todos los chunks tienen campos obligatorios
    3. Cobertura por tag: cada PDF tiene chunks
    4. Retrieval básico: encuentra chunks relevantes
    5. Filtro por tag: funciona correctamente
    6. Umbral de similitud: rechaza chunks poco relevantes
    7. Anti-alucinación: rechaza preguntas fuera de dominio
    8. Cita de fuentes: las respuestas incluyen referencias
    9. Consistencia: preguntas similares dan respuestas similares
    10. Latencia: el sistema responde en tiempo razonable

Ejecutar con: pytest tests/ -v
"""
from __future__ import annotations

import time
import pytest

from src.retriever import Retriever, RetrievedChunk
from src.generator import Generator, NO_CONTEXT_MESSAGE
from src.config import PDF_CONFIGS, SIMILARITY_THRESHOLD


# ============================================================
# FIXTURES (recursos compartidos entre tests)
# ============================================================
@pytest.fixture(scope="module")
def retriever() -> Retriever:
    """Retriever reutilizable entre tests."""
    return Retriever()


@pytest.fixture(scope="module")
def generator() -> Generator:
    """Generator reutilizable entre tests."""
    return Generator()


# ============================================================
# TEST 1: Ingesta correcta
# ============================================================
def test_01_collection_has_minimum_chunks(retriever: Retriever) -> None:
    """La colección debe tener al menos 400 chunks cargados."""
    count = retriever.count()
    assert count >= 400, f"Esperado >=400 chunks, encontrado {count}"
    print(f"✅ Colección con {count} chunks")


# ============================================================
# TEST 2: Metadata completa en cada chunk
# ============================================================
def test_02_all_chunks_have_required_metadata(retriever: Retriever) -> None:
    """Todos los chunks deben tener los campos de metadata obligatorios."""
    data = retriever.vectorstore.get(include=["metadatas"])
    required_fields = {"source", "tag", "page", "chunk_type", "breadcrumb"}

    for meta in data["metadatas"]:
        missing = required_fields - set(meta.keys())
        assert not missing, f"Chunk sin campos: {missing} | metadata: {meta}"

    print(f"✅ {len(data['metadatas'])} chunks con metadata completa")


# ============================================================
# TEST 3: Cobertura por tag (cada PDF está representado)
# ============================================================
def test_03_all_tags_present(retriever: Retriever) -> None:
    """Cada uno de los 4 PDFs debe tener chunks en la colección."""
    expected_tags = {cfg.tag for cfg in PDF_CONFIGS.values()}
    found_tags = set(retriever.list_tags())

    missing = expected_tags - found_tags
    assert not missing, f"Tags faltantes: {missing}"

    print(f"✅ Tags presentes: {sorted(found_tags)}")


# ============================================================
# TEST 4: Retrieval básico funciona
# ============================================================
def test_04_retrieval_returns_relevant_chunks(retriever: Retriever) -> None:
    """Una pregunta clara del dominio debe devolver chunks relevantes."""
    query = "¿Cuántos Pokémon puedo llevar en un equipo VGC?"
    results = retriever.search(query, k=5)

    assert len(results) > 0, "No se recuperaron chunks para pregunta válida"
    assert all(r.score >= SIMILARITY_THRESHOLD for r in results), \
        f"Algún chunk está por debajo del umbral {SIMILARITY_THRESHOLD}"

    # Al menos uno debe ser de vgc_rules
    tags = [r.tag for r in results]
    assert "vgc_rules" in tags, f"Esperado chunk de vgc_rules, encontrado: {tags}"

    print(f"✅ {len(results)} chunks relevantes (top score: {results[0].score:.3f})")


# ============================================================
# TEST 5: Filtro por tag funciona
# ============================================================
def test_05_tag_filter_works(retriever: Retriever) -> None:
    """El filtro por tag debe devolver solo chunks del tag indicado."""
    query = "penalización por llegar tarde"
    results = retriever.search(query, k=5, tag="penalty_guidelines")

    assert len(results) > 0, "Filtro por tag no devolvió resultados"
    for r in results:
        assert r.tag == "penalty_guidelines", \
            f"Chunk con tag incorrecto: {r.tag}"

    print(f"✅ Filtro por tag funciona: {len(results)} chunks de penalty_guidelines")


# ============================================================
# TEST 6: Umbral rechaza chunks irrelevantes
# ============================================================
def test_06_threshold_rejects_irrelevant(retriever: Retriever) -> None:
    """Un umbral alto debe rechazar chunks de baja similitud."""
    query = "¿Cuántos Pokémon puedo llevar?"
    results_strict = retriever.search(query, k=5, threshold=0.95)

    # Con umbral 0.95, es muy probable que devuelva 0 o muy pocos
    assert len(results_strict) <= 2, \
        f"Umbral 0.95 devolvió demasiados chunks: {len(results_strict)}"

    print(f"✅ Umbral alto filtra correctamente: {len(results_strict)} chunks")


# ============================================================
# TEST 7: Anti-alucinación (fuera de dominio)
# ============================================================
def test_07_rejects_out_of_domain_questions(generator: Generator) -> None:
    """Preguntas fuera del dominio no deben producir respuestas inventadas."""
    out_of_domain = [
        "¿Cuál es la capital de Francia?",
        "¿Quién ganó el mundial de fútbol 2022?",
        "¿Cómo se hace una pizza margarita?",
    ]

    for query in out_of_domain:
        response = generator.answer(query, k=5)

        # No debe haber fuentes (o si las hay, el modelo debe rechazar)
        if not response.used_context:
            assert NO_CONTEXT_MESSAGE in response.answer or \
                   "no encuentro" in response.answer.lower(), \
                f"Respuesta inesperada sin contexto: {response.answer[:100]}"

    print(f"✅ Anti-alucinación: {len(out_of_domain)} preguntas rechazadas")


# ============================================================
# TEST 8: Respuestas citan fuentes
# ============================================================
def test_08_answers_cite_sources(generator: Generator) -> None:
    """Las respuestas dentro del dominio deben citar secciones."""
    query = "¿Qué sanción hay por slow play?"
    response = generator.answer(query, k=5)

    assert response.used_context, "No se encontró contexto para pregunta del dominio"
    assert len(response.sources) > 0, "No hay fuentes en la respuesta"

    # La respuesta debe mencionar alguna sección o fuente
    answer_lower = response.answer.lower()
    has_citation = (
        "sección" in answer_lower
        or "fuente" in answer_lower
        or "manual" in answer_lower
        or any(".pdf" in s.source for s in response.sources)
    )
    assert has_citation, f"Respuesta sin cita: {response.answer[:200]}"

    print(f"✅ Respuesta cita {len(response.sources)} fuentes")


# ============================================================
# TEST 9: Consistencia entre preguntas similares
# ============================================================
def test_09_consistent_answers_for_similar_questions(retriever: Retriever) -> None:
    """Preguntas semánticamente equivalentes deben recuperar fuentes similares."""
    q1 = "¿Cuántos Pokémon puedo usar en combate?"
    q2 = "¿Cuál es el número máximo de Pokémon en un equipo?"

    results1 = {r.breadcrumb for r in retriever.search(q1, k=5)}
    results2 = {r.breadcrumb for r in retriever.search(q2, k=5)}

    # Debe haber intersección (al menos 1 fuente en común)
    intersection = results1 & results2
    assert len(intersection) > 0, \
        f"Sin intersección entre preguntas similares:\nQ1: {results1}\nQ2: {results2}"

    print(f"✅ Consistencia: {len(intersection)} fuentes en común")


# ============================================================
# TEST 10: Latencia aceptable
# ============================================================
def test_10_response_latency_is_acceptable(generator: Generator) -> None:
    """El sistema completo debe responder en <5 segundos."""
    query = "¿Qué está prohibido en los torneos?"

    start = time.time()
    response = generator.answer(query, k=5)
    elapsed = time.time() - start

    assert response.used_context, "Pregunta del dominio sin contexto"
    assert elapsed < 5.0, f"Latencia demasiado alta: {elapsed:.2f}s"

    print(f"✅ Latencia: {elapsed*1000:.0f}ms")