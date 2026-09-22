Para entender cómo se construye un prototipo RAG de punta a punta sin perderse en teoría abstracta, lo más efectivo es analizar su anatomía básica en código.

  [(265) Crea tu Propio RAG con LangChain | HABLA con tus PDFs 📖 - YouTube](https://www.youtube.com/watch?v=0MAILan8Agw)

A continuación tienes la explicación paso a paso de cómo funciona el flujo, los términos exactos recomendados para buscar tutoriales en video y un ejemplo mínimo funcional en Python que pueden usar como plantilla inicial.

  

### 1. El Flujo de un Prototipo RAG en 4 Pasos

1. **Carga y Fragmentación (_Chunking_):**
    
      
    - Se lee el PDF del reglamento (por ejemplo, el PDF de reglas de Pokémon VGC).
        
          
        
    - Se divide en bloques pequeños (de unos 500 a 800 caracteres) con un leve traslape (_overlap_) para que las oraciones no queden cortadas a la mitad.
        
          
        
2. **Vectorización (_Embeddings_) y Almacenamiento:**
    
      
    - Cada fragmento se envía a un modelo de embeddings, que convierte el texto en una lista de números (un vector que representa su significado).
        
          
        
    - Estos vectores y el texto original se guardan en una base de datos vectorial local (como **ChromaDB**).
        
          
        
3. **Recuperación (_Retrieval_):**
    
      
    - El usuario pregunta: _"¿Cuántos minutos dura el turno de combate?"_.
        
          
        
    - La pregunta se convierte a vector y ChromaDB busca los 2 o 3 fragmentos de texto más parecidos matemáticamente (por similitud de coseno).
        
          
        
4. **Aumento y Generación (_Augmentation & Generation_):**
    
      
    - Se construye un _prompt_ que combina la pregunta con los fragmentos recuperados.
        
          
        
    - Se le ordena al LLM: _"Usa solo el siguiente contexto para responder. Si no está en el contexto, di que no lo sabes. Cita la sección/página"_.
        
          
        
    - El LLM genera la respuesta con su respectiva cita.
        
          
        

### 2. Términos de Búsqueda Recomendados para Videos (YouTube)

Para ver el paso a paso en pantalla con código real, busca cualquiera de estos términos:

  

- **"RAG desde cero con LangChain y ChromaDB en Python"** _(ideal para ver la estructura modular)._
    
      
    
- **"Construye un RAG para consultar tus propios PDFs con Python"** _(el caso exacto que van a resolver)._
    
      
    
- **"RAG minimalista con Streamlit y Chroma"** _(muestra cómo armar la interfaz visual en pocas líneas para la demo)._
    
      
    
- **"Python RAG tutorial citations and source tracking"** _(útil para ver cómo obligar al sistema a devolver el número de página o documento)._
    
      
    

### 3. Ejemplo Mínimo Funcional (Mental Model en Código)

Este es un ejemplo simplificado de la lógica que implementaría el **Integrante 3** (Ingeniero de IA/Datos) en el backend:

  

Python

```
# 1. Cargar y partir el documento
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Cargamos el reglamento oficial de Pokémon
loader = PyPDFLoader("reglamento_vgc_2026.pdf")
docs = loader.load()

# Partimos el texto en bloques con contexto continuo
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# 2. Generar Embeddings y guardar en base vectorial local
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings # o OpenAIEmbeddings

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    persist_directory="./chroma_db"
)

# 3. Función de consulta (Retriever)
def responder_consulta_reglamento(pregunta_usuario: str):
    # Recupera los 3 fragmentos más relevantes
    docs_relevantes = vector_db.similarity_search(pregunta_usuario, k=3)
    
    # Preparamos el contexto y las fuentes encontradas
    contexto = "\n---\n".join([d.page_content for d in docs_relevantes])
    fuentes = [f"Página {d.metadata.get('page', 'desconocida')}" for d in docs_relevantes]
    
    # 4. Prompt estricto anti-alucinaciones
    prompt = f"""
    Eres un asistente árbitro del torneo oficial de Pokémon.
    Responde a la siguiente consulta utilizando ÚNICAMENTE la información provista en el contexto.
    Si la información no aparece explícitamente en el contexto, di textualmente:
    "Esta regla no se encuentra en la documentación oficial provista."
    Al responder, indica claramente de qué sección o página proviene la regla.

    Contexto oficial:
    {contexto}

    Pregunta del usuario:
    {pregunta_usuario}
    """
    
    # Aquí se llama al LLM (ej. Gemini o GPT) pasándole el prompt
    # return respuesta_del_modelo, fuentes
```

### 4. ¿Qué debe verse en la pantalla durante la Demo?

Para que la demo del **Hito 2** y de la **Presentación Final** convenza al catedrático de que no es un simple ChatGPT disfrazado, la interfaz (que diseñará el **Integrante 5**) debe mostrar tres cosas claras:

  

1. **La caja de chat:** Donde el usuario escribe la pregunta.
    
      
    
2. **La respuesta sintetizada:** La explicación en lenguaje natural generada por el modelo.
    
      
    
3. **El acordeón o tarjeta de "Fuentes consultadas":** Una sección desplegable abajo de la respuesta donde se lea:
    
      
    
    > _Fuente: Reglamento_VGC.pdf — Página 14, Párrafo 2:_
    > 
    >   
    > 
    > _"Los combates tendrán un límite de tiempo por turno de 45 segundos por jugador..."_
    > 
    >   
    

Al mostrar el texto original del PDF al lado de la respuesta, demuestran en vivo que el sistema está extrayendo información real mediante RAG y citando las fuentes correctamente.