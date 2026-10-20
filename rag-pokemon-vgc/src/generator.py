"""
Generator: construcción de prompt estricto + llamada a Groq.
Incluye retry, logging y manejo robusto de errores.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field

from groq import Groq, APIError, RateLimitError, APIConnectionError

from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    TOP_K,
)
from src.retriever import Retriever, RetrievedChunk, format_retrieved_for_context

logger = logging.getLogger(__name__)


# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """Eres un asistente experto en las reglas oficiales de los torneos de Pokémon VGC (Video Game Championships) y Play! Pokémon.

Tu única fuente de verdad es el CONTEXTO proporcionado, extraído de los manuales oficiales.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE basándote en el CONTEXTO. No uses conocimiento previo ni inventes información.
2. Si el CONTEXTO no contiene la respuesta, di exactamente: "No encuentro esa información en los manuales oficiales que tengo disponibles." No intentes adivinar.
3. Cuando cites una regla, menciona la sección (por ejemplo: "según la sección 4.2 del manual...") y la página si está disponible.
4. Si la pregunta es ambigua, pide aclaración antes de responder.
5. Responde SIEMPRE en español, con lenguaje claro y directo.
6. Si el contexto contiene tablas, respétalas: los valores numéricos y las penalizaciones son exactos.
7. No inventes números de sección, páginas ni citas textuales. Solo cita lo que aparece explícitamente en el contexto.

FORMATO DE RESPUESTA:
- Respuesta directa primero (1-3 frases).
- Detalles o matices después, si aplica.
- Fuentes consultadas al final, en una línea con formato:
  "📚 Fuentes: <archivo1> (sección X.Y), <archivo2> (sección Z.W)"
"""

USER_PROMPT_TEMPLATE = """CONTEXTO:
{context}

PREGUNTA DEL USUARIO:
{question}

Responde siguiendo estrictamente las reglas del sistema."""

NO_CONTEXT_MESSAGE = (
    "No encuentro esa información en los manuales oficiales que tengo disponibles."
)


# ============================================================
# RESULTADO
# ============================================================
@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk]
    query: str
    model: str
    used_context: bool
    tokens_used: int = 0
    latency_ms: float = 0.0

    @property
    def source_summary(self) -> str:
        if not self.sources:
            return "Sin fuentes"
        seen = set()
        parts = []
        for s in self.sources:
            key = (s.source, s.breadcrumb)
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"{s.source} — {s.breadcrumb or 'N/A'}")
        return " | ".join(parts)


# ============================================================
# GENERATOR
# ============================================================
class Generator:
    def __init__(self) -> None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY no configurada en .env")

        self.client = Groq(api_key=GROQ_API_KEY)
        self.retriever = Retriever()
        logger.info(f"Generator listo. Modelo: {LLM_MODEL}")

    def _call_llm(
        self,
        messages: list[dict],
        max_retries: int = 3,
    ) -> tuple[str, int]:
        """
        Llama a Groq con reintentos y manejo de errores.
        Devuelve (texto, tokens_usados).
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )
                text = completion.choices[0].message.content or ""
                tokens = completion.usage.total_tokens if completion.usage else 0
                return text.strip(), tokens

            except RateLimitError as e:
                wait = 2 ** attempt
                logger.warning(f"Rate limit (intento {attempt}/{max_retries}). "
                               f"Esperando {wait}s...")
                last_error = e
                time.sleep(wait)

            except APIConnectionError as e:
                wait = 2 ** attempt
                logger.warning(f"Error de conexión (intento {attempt}/{max_retries}). "
                               f"Esperando {wait}s...")
                last_error = e
                time.sleep(wait)

            except APIError as e:
                # Errores 4xx (modelo no existe, key inválida, etc.)
                if e.status_code == 404:
                    raise RuntimeError(
                        f"El modelo '{LLM_MODEL}' no está disponible con tu API key. "
                        f"Ejecuta scripts/list_groq_models.py para ver los disponibles."
                    ) from e
                if e.status_code == 401:
                    raise RuntimeError(
                        "API key inválida. Verifica GROQ_API_KEY en .env"
                    ) from e
                last_error = e
                time.sleep(2)
            except Exception as e:
                last_error = e
                time.sleep(2)

        raise RuntimeError(
            f"LLM falló tras {max_retries} intentos. Último error: {last_error}"
        )

    def answer(
        self,
        question: str,
        k: int = TOP_K,
        tag: str | None = None,
    ) -> RAGResponse:
        """Flujo completo: retrieve → prompt → LLM → respuesta con fuentes."""
        t0 = time.time()

        # 1) Retrieval
        chunks = self.retriever.search(question, k=k, tag=tag)
        used_context = len(chunks) > 0

        # 2) Short-circuit: sin contexto, no llamamos al LLM
        if not used_context:
            return RAGResponse(
                answer=NO_CONTEXT_MESSAGE,
                sources=[],
                query=question,
                model=LLM_MODEL,
                used_context=False,
                latency_ms=(time.time() - t0) * 1000,
            )

        # 3) Construir prompt
        context = format_retrieved_for_context(chunks)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

        # 4) Llamar al LLM
        try:
            answer_text, tokens = self._call_llm([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ])
        except Exception as e:
            answer_text = f"❌ Error al consultar el LLM: {e}"
            tokens = 0

        return RAGResponse(
            answer=answer_text,
            sources=chunks,
            query=question,
            model=LLM_MODEL,
            used_context=used_context,
            tokens_used=tokens,
            latency_ms=(time.time() - t0) * 1000,
        )

    def answer_stream(self, question: str, k: int = TOP_K, tag: str | None = None):
        """Versión streaming (útil para Streamlit)."""
        chunks = self.retriever.search(question, k=k, tag=tag)

        if not chunks:
            yield NO_CONTEXT_MESSAGE
            return

        context = format_retrieved_for_context(chunks)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

        stream = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True,
        )

        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🔌 Inicializando Generator...")
    gen = Generator()

    test_questions = [
        "¿Cuántos Pokémon puedo llevar en un equipo VGC?",
        "¿Qué pasa si llego tarde a una ronda de torneo?",
        "¿Puedo usar un Pokémon shiny?",
        "¿Cómo se reporta una conducta inapropiada de otro jugador?",
        "¿Cuál es la capital de Francia?",
    ]

    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"❓ {q}")
        response = gen.answer(q, k=3)
        print(f"💬 {response.answer}")
        print(f"📚 {response.source_summary}")
        print(f"⚡ {response.latency_ms:.0f}ms | 🪙 {response.tokens_used} tokens")