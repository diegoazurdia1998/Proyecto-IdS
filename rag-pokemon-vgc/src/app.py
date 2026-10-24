"""
Interfaz Streamlit para el RAG de reglas Pokémon VGC.
Ejecutar con: streamlit run src/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Configuración de la página (debe ir primero)
st.set_page_config(
    page_title="Pokémon VGC RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import (
    LLM_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD,
    CHROMA_COLLECTION_NAME,
)
from src.retriever import Retriever, RetrievedChunk
from src.generator import Generator, NO_CONTEXT_MESSAGE


# ============================================================
# ESTILOS PERSONALIZADOS
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FFCB05 0%, #3B4CCA 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
    }
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #3B4CCA;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .source-card .breadcrumb {
        color: #3B4CCA;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .source-card .meta {
        color: #666;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }
    .score-badge {
        display: inline-block;
        background: #28a745;
        color: white;
        padding: 0.1rem 0.5rem;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .no-context {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHE DE RECURSOS PESADOS
# ============================================================
@st.cache_resource(show_spinner="🔄 Cargando modelo de embeddings...")
def get_generator() -> Generator:
    """Inicializa el Generator una sola vez (cacheado)."""
    return Generator()


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar() -> dict:
    """Renderiza la barra lateral con opciones de configuración."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")

        st.markdown(f"**Modelo LLM:** `{LLM_MODEL}`")
        st.markdown(f"**Colección:** `{CHROMA_COLLECTION_NAME}`")
        st.divider()

        # Filtro por tag
        st.markdown("### 🏷️ Filtrar por documento")
        tag_options = {
            "Todos los documentos": None,
            "📖 Reglas VGC": "vgc_rules",
            "📜 Reglas generales": "general_rules",
            "⚖️ Penalizaciones": "penalty_guidelines",
            "🤝 Normas de conducta": "standards_conduct",
        }
        tag_label = st.selectbox(
            "Buscar en:",
            options=list(tag_options.keys()),
            index=0,
        )
        selected_tag = tag_options[tag_label]

        st.divider()

        # Parámetros de retrieval
        st.markdown("### 🔍 Búsqueda")
        k = st.slider(
            "Chunks a recuperar (k)",
            min_value=1,
            max_value=10,
            value=TOP_K,
            help="Cuántos fragmentos buscar en la base vectorial.",
        )
        threshold = st.slider(
            "Umbral de similitud",
            min_value=0.5,
            max_value=0.95,
            value=SIMILARITY_THRESHOLD,
            step=0.01,
            help="Similitud mínima para aceptar un chunk como relevante.",
        )

        st.divider()

        # Info
        st.markdown("### ℹ️ Acerca de")
        st.caption(
            "RAG basado en los manuales oficiales de Play! Pokémon. "
            "Las respuestas se basan **exclusivamente** en los documentos cargados."
        )

        # Botón limpiar historial
        if st.button("🗑️ Limpiar conversación", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        return {"tag": selected_tag, "k": k, "threshold": threshold}


# ============================================================
# RENDER DE FUENTES
# ============================================================
def render_sources(sources: list[RetrievedChunk]) -> None:
    """Muestra los chunks recuperados de forma visual."""
    if not sources:
        return

    with st.expander(f"📚 Fuentes consultadas ({len(sources)} chunks)", expanded=False):
        for i, src in enumerate(sources, start=1):
            st.markdown(f"""
            <div class="source-card">
                <div class="breadcrumb">[{i}] {src.breadcrumb or 'Sin sección'}</div>
                <div class="meta">
                    📄 {src.source} &nbsp;·&nbsp;
                    Pág. {src.page} &nbsp;·&nbsp;
                    Tipo: {src.chunk_type} &nbsp;·&nbsp;
                    <span class="score-badge">score {src.score:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.container():
                st.caption("Contenido del chunk:")
                st.code(src.content[:500] + ("..." if len(src.content) > 500 else ""),
                        language="markdown")


# ============================================================
# APP PRINCIPAL
# ============================================================
def main() -> None:
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Pokémon VGC RAG</h1>
        <p>Asistente inteligente sobre las reglas oficiales de torneos Pokémon</p>
    </div>
    """, unsafe_allow_html=True)

    # Inicializar historial
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    config = render_sidebar()

    # Verificar que el Generator esté listo
    try:
        generator = get_generator()
    except Exception as e:
        st.error(f"❌ Error al inicializar el RAG: {e}")
        st.info(
            "Verifica que:\n"
            "1. `GROQ_API_KEY` esté configurada en `.env`\n"
            "2. Hayas corrido `python -m src.ingestion`\n"
            "3. El modelo en `.env` sea válido para tu cuenta de Groq"
        )
        return

    # Mostrar historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    # Input del usuario
    if prompt := st.chat_input("Pregunta sobre las reglas de Pokémon VGC..."):
        # Añadir mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            start_time = time.time()

            # 1) Retrieval manual con los parámetros del sidebar
            try:
                sources = generator.retriever.search(
                    query=prompt,
                    k=config["k"],
                    tag=config["tag"],
                    threshold=config["threshold"],
                )
            except Exception as e:
                st.error(f"❌ Error en retrieval: {e}")
                return

            # 2) Short-circuit si no hay contexto
            if not sources:
                st.markdown(f"""
                <div class="no-context">
                    <strong>⚠️ Sin contexto relevante</strong><br>
                    {NO_CONTEXT_MESSAGE}
                </div>
                """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": NO_CONTEXT_MESSAGE,
                    "sources": [],
                })
                return

            # 3) Llamar al LLM con streaming
            from src.retriever import format_retrieved_for_context
            from src.generator import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

            context = format_retrieved_for_context(sources)
            user_prompt = USER_PROMPT_TEMPLATE.format(
                context=context, question=prompt
            )

            try:
                with st.spinner("💭 Pensando..."):
                    stream = generator.client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                        max_tokens=1024,
                        stream=True,
                    )

                # Streaming visual
                response_placeholder = st.empty()
                full_response = ""
                for chunk_event in stream:
                    delta = chunk_event.choices[0].delta.content
                    if delta:
                        full_response += delta
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"❌ Error al consultar el LLM: {e}")
                return

            # 4) Render de fuentes
            render_sources(sources)

            # 5) Footer con métricas
            elapsed = time.time() - start_time
            st.caption(
                f"⚡ {elapsed*1000:.0f}ms · "
                f"📚 {len(sources)} chunks · "
                f"🤖 {LLM_MODEL}"
            )

            # 6) Guardar en historial
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })


if __name__ == "__main__":
    main()