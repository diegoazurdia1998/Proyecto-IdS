# ⚡ Pokémon VGC RAG
> Sistema de Generación Aumentada por Recuperación (RAG) para consultar las reglas oficiales de los torneos de Pokémon VGC (Video Game Championships) y Play! Pokémon.
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-red.svg)](https://groq.com/)
---
## 📋 Descripción
Este proyecto implementa un **RAG (Retrieval-Augmented Generation)** que responde preguntas en español sobre las reglas oficiales de torneos Pokémon, basándose **exclusivamente** en los manuales oficiales de Play! Pokémon.
### Características
- ✅ **Anti-alucinaciones**: responde solo con base en los documentos cargados
- ✅ **Cita de fuentes**: cada respuesta incluye sección, página y archivo
- ✅ **Filtrado por documento**: búsqueda dirigida a un manual específico
- ✅ **Parsing jerárquico**: preserva la estructura de secciones y tablas
- ✅ **Embeddings multilingües**: excelente desempeño en español
- ✅ **Streaming**: respuestas token a token en la interfaz
- ✅ **100% gratuito**: embeddings locales + LLM vía Groq free tier
---
## 🏗️ Arquitectura
```text

┌─────────────────────────────────────────────────────────────┐  
│ PDFs Oficiales                                              │  
│ (4 manuales de Play! Pokémon — ~140 páginas en total)       │  
└───────────────────────────┬─────────────────────────────────┘  
							│  
							▼  
┌─────────────────────────────────────────────────────────────┐  
│ INGESTA (src/ingestion.py)                                  │  
│ • PyMuPDF: extracción de texto + detección de headers       │  
│ • pdfplumber: extracción de tablas                          │  
│ • Chunking jerárquico con breadcrumbs                       │  
│ • Metadata enriquecida (tag, sección, página, tipo)         │  
└───────────────────────────┬─────────────────────────────────┘  
							│  
							▼  
┌─────────────────────────────────────────────────────────────┐  
│ EMBEDDINGS (multilingual-e5-base)                           │  
│ • Modelo local (768 dim) — gratis y sin API                 │  
│ • Prefijos E5: "passage:" para docs, "query:" para queries  │  
└───────────────────────────┬─────────────────────────────────┘  
                            │  
							▼  
┌─────────────────────────────────────────────────────────────┐  
│ CHROMADB (440 chunks)                                       │  
│ • Base vectorial local persistente                          │  
│ • Filtrado por metadata (tag, chunk_type)                   │  
└───────────────────────────┬─────────────────────────────────┘  
							│  
							▼  
┌─────────────────────────────────────────────────────────────┐  
│ RETRIEVAL (src/retriever.py) + GENERATOR                    │  
│ • Búsqueda semántica con umbral de similitud                │  
│ • Prompt estricto anti-alucinaciones                        │  
│ • LLM: Groq (openai/gpt-oss-120b)                           │  
└───────────────────────────┬─────────────────────────────────┘  
							│  
							▼  
┌─────────────────────────────────────────────────────────────┐  
│ INTERFAZ (src/app.py — Streamlit)                           │  
│ • Chat con streaming                                        │  
│ • Filtro por documento                                      |  
│ • Visualización de fuentes con scores                       │  
└─────────────────────────────────────────────────────────────┘
```

---
## 📁 Estructura del Proyecto

rag-pokemon-vgc/  
├── data/                                        # Corpus documental (4 PDFs)  
├── chroma_db/                            # Base vectorial (auto-generada)  
├── preprocess/                            # Scripts de análisis previo  
│ ├── analyze_pdfs.py  
│ ├── inspect_structure.py  
│ ├── pdf_analysis_report.json  
│ ├── structure_report.json  
│ └── ingestion_summary.json  
├── src/  
│ ├── **init**.py  
│ ├── config.py                              # Configuración central  
│ ├── ingestion.py # Pipeline de ingesta  
│ ├── retriever.py # Búsqueda semántica  
│ ├── generator.py # LLM + prompt  
│ └── app.py # Interfaz Streamlit  
├── tests/  
│ └── test_rag.py                           # 10 tests del Plan de Calidad  
├── scripts/  
│ └── list_groq_models.py # Utility: verificar modelos  
├── .env.example  
├── .gitignore  
├── pytest.ini  
├── conftest.py  
├── requirements.txt  
└── README.md


---
## 🚀 Instalación y Uso
### Requisitos previos
- **Python 3.11** o superior (recomendado 3.11/3.12)
- **Cuenta en Groq** (gratis) → https://console.groq.com/keys
- **~2 GB de espacio** en disco (para el modelo de embeddings)
### Paso 1: Clonar e instalar

```bash
git clone <tu-repo>
cd rag-pokemon-vgc
# Crear entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

### Paso 2: Configurar API key

```bash

cp .env.example .env
```

Edita `.env` y añade tu key de Groq:

```env

GROQ_API_KEY=gsk_tu_key_aqui
LLM_MODEL=openai/gpt-oss-120b
```
### Paso 3: Descargar los PDFs

Los 4 PDFs oficiales deben estar en `data/`:

```bash

data/
├── play-pokemon-penalty-guidelines-es.pdf
├── play-pokemon-standards-of-conduct-es.pdf
├── play-pokemon-tournament-rules-handbook-es.pdf
└── play-pokemon-vgc-tournament-handbook-es.pdf
```
### Paso 4: Ingesta de documentos

```bash

python -m src.ingestion
```
⏳ **La primera vez tarda 5-10 min** (descarga modelo + embeddings en CPU).

**Salida esperada:**

```text

📊 Total de chunks: 440
   Distribución por tag:
     - penalty_guidelines: 167
     - general_rules: 161
     - vgc_rules: 64
     - standards_conduct: 48
✅ Cargados 440 chunks en ChromaDB
```
### Paso 5: Ejecutar la aplicación

```bash

streamlit run src/app.py
```
Se abrirá en `http://localhost:8501`.

---

## 🧪 Tests

```bash
pytest
```
**Salida esperada:**

```text

======== 10 passed in 17.64s ========
```
Los 10 tests cubren:

|#|Test|Qué valida|
|---|---|---|
|1|`collection_has_minimum_chunks`|Ingesta correcta (≥400 chunks)|
|2|`all_chunks_have_required_metadata`|Metadata completa|
|3|`all_tags_present`|Cobertura de los 4 PDFs|
|4|`retrieval_returns_relevant_chunks`|Retrieval funcional|
|5|`tag_filter_works`|Filtrado por tag|
|6|`threshold_rejects_irrelevant`|Umbral de similitud|
|7|`rejects_out_of_domain_questions`|Anti-alucinación|
|8|`answers_cite_sources`|Trazabilidad|
|9|`consistent_answers_for_similar_questions`|Consistencia|
|10|`response_latency_is_acceptable`|Latencia <5s|

---

## ⚙️ Configuración avanzada

Todos los parámetros están en `.env` y `src/config.py`:

|Variable|Descripción|Default|
|---|---|---|
|`GROQ_API_KEY`|API key de Groq (obligatorio)|—|
|`LLM_MODEL`|Modelo de Groq|`openai/gpt-oss-120b`|
|`LLM_TEMPERATURE`|Temperatura del LLM|`0.1`|
|`LLM_MAX_TOKENS`|Máx. tokens por respuesta|`1024`|
|`EMBEDDING_MODEL`|Modelo de embeddings|`intfloat/multilingual-e5-base`|
|`CHROMA_COLLECTION_NAME`|Nombre de la colección|`pokemon_vgc_rules`|
|`TOP_K`|Chunks recuperados|`5`|
|`SIMILARITY_THRESHOLD`|Umbral de similitud|`0.7`|

### Estrategia de chunking por PDF

Cada PDF tiene una configuración optimizada basada en su estructura:

|PDF|chunk_size|overlap|Estrategia|
|---|---|---|---|
|`penalty-guidelines`|600|100|**Por fila de tabla** (matriz infracción→sanción)|
|`standards-of-conduct`|700|120|Jerárquico|
|`tournament-rules-handbook`|800|150|Jerárquico + tablas|
|`vgc-tournament-handbook`|700|130|Jerárquico + tablas|

---

## 🔍 Cómo funciona el anti-alucinaciones

El sistema tiene **3 capas de defensa**:

1. **Filtro de similitud en retrieval**: solo recupera chunks con score ≥ 0.7. Si nada supera el umbral, no se llama al LLM.
    
2. **Short-circuit**: si no hay chunks relevantes, devuelve directamente `"No encuentro esa información en los manuales oficiales"` sin gastar tokens.
    
3. **System prompt estricto**: instruye al LLM a:
    
    - Responder solo con base en el contexto
        
    - Decir "no encuentro" si no está en el contexto
        
    - No inventar secciones, páginas ni citas
        

### Ejemplo de validación

|Pregunta|Comportamiento esperado|
|---|---|
|_"¿Cuántos Pokémon puedo llevar?"_|✅ Responde citando sección 2.1|
|_"¿Qué sanción hay por slow play?"_|✅ Responde citando tabla|
|_"¿Cuál es la capital de Francia?"_|✅ **Rechaza** (sin fuentes)|

---

## 📊 Métricas del sistema

|Métrica|Valor|
|---|---|
|**Documentos**|4 PDFs oficiales|
|**Páginas totales**|~140|
|**Chunks generados**|440|
|**Chunks de texto**|263 (60%)|
|**Chunks de tabla**|177 (40%)|
|**Modelo de embeddings**|multilingual-e5-base (768d, local)|
|**Latencia promedio**|~1s (incluye retrieval + LLM)|
|**Costo por consulta**|$0.00 (Groq free tier)|
|**Cobertura de tests**|10/10 ✅|

---

## 🛠️ Tecnologías utilizadas

- **PyMuPDF (fitz)** — Extracción de texto y análisis de layout
    
- **pdfplumber** — Extracción de tablas
    
- **LangChain** — Orquestación del RAG
    
- **ChromaDB** — Base de datos vectorial
    
- **sentence-transformers** — Embeddings locales
    
- **Groq** — LLM (gratis, ultra-rápido)
    
- **Streamlit** — Interfaz web
    
- **pytest** — Testing
    

---

## 🎯 Casos de uso

- **Jugadores**: consultar reglas antes de un torneo
    
- **Jueces**: verificar sanciones aplicables
    
- **Organizadores**: resolver dudas sobre formatos
    
- **Aprendizaje**: entender las reglas oficiales de forma conversacional
    

---

## 👥 Autor

**[Tu nombre]**  
Proyecto de Ciencia de Datos — Ingeniería de Software  
Semestre 18 — [Tu Universidad]