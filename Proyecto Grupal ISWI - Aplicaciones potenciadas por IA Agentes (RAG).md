

## 📌 Metadatos del Proyecto
- **Curso:** Ingeniería de Software I — 2do Ciclo 2026
- **Tema Asignado:** Aplicaciones potenciadas por IA / agentes (RAG)
- **Modalidad:** Grupal (Enfoque en rebanada vertical / *end-to-end*)
- **Estado:** 🟡 En definición de problema

---

## 📅 Cronograma e Hitos Críticos
- [ ] **Hito 1 (12/10/2026):** Problema elegido + Mini-SRS + Diagrama de arquitectura C4.
- [ ] **Hito 2 (21/10/2026):** Ensayo general (*dry-run*) de la demo del prototipo funcionando.
- [ ] **Entrega Final y Presentaciones (02/11/2026 y 04/11/2026):** Subida al portal + exposición en clase (20-25 min).

---

## 🎯 Alcance Específico del Tema (RAG / Agentes)

### ⚠️ Lo que DEBE incluir el prototipo (Rebanada Vertical)
El proyecto exige profundidad y funcionalidad real sobre una sola rebanada vertical:
- **Opción A (RAG completo):**
  - Ingesta y preprocesamiento de documentos (chunking, metadatos).
  - Base de datos vectorial (*vector store*) para almacenamiento y búsqueda semántica.
  - Mecanismo de consulta que recupere contexto relevante.
  - Generación de respuesta **citando fuentes y controlando alucinaciones**.
- **Opción B (Agente con herramientas):**
  - Agente de IA que razone e invoque de manera autónoma al menos **1 herramienta real** (API, consulta a base de datos, script, etc.).

> 🚫 **Criterio de Rechazo Explícito:** Un chatbot que solo reenvía prompts directamente a un LLM sin recuperación contextual, sin base vectorial y sin control de alucinaciones **NO cuenta como prototipo**.

---

## 📋 Matriz de Entregables Obligatorios (Camino a los 100 pts)

### 1. Mini-SRS (Requisitos de Software)
- [ ] 3 a 5 requisitos funcionales claramente delimitados.
- [ ] 1 a 2 casos de uso o historias de usuario que acoten estrictamente el alcance.

### 2. Arquitectura de Software
- [ ] Diagrama de arquitectura en modelo **C4 (Nivel Contenedor)**.
- [ ] Justificación explícita de **trade-offs** arquitectónicos (por qué esta solución frente a alternativas como un LLM puro, fine-tuning, monolith vs microservicio, etc.).

### 3. Plan de Calidad
- [ ] Selección de **2 o 3 atributos de calidad ISO/IEC 25010** priorizados para RAG (ej. exactitud/precisión, latencia/rendimiento, fiabilidad, mantenibilidad).
- [ ] Estrategia y diseño de pruebas concretas para validar dichos atributos (ej. pruebas de recuperación, evaluación de alucinaciones con Ragas/TruLens, pruebas de carga).

### 4. Análisis de Seguridad
- [ ] Aplicación de la **Tríada CIA** (Confidencialidad, Integridad, Disponibilidad) aplicada al dominio elegido.
- [ ] Identificación de **2 o 3 riesgos OWASP** pertinentes (ej. *OWASP Top 10 for LLMs*: Prompt Injection, Insecure Output Handling, Training Data Poisoning / Context Leaks) junto a su plan de mitigación.

### 5. Repositorio, Código y Gestión Kanban
- [ ] Repositorio con historial de commits activo y colaborativo de todos los miembros.
- [ ] Tablero **Kanban** actualizado que refleje la división equitativa de tareas de cada integrante.
- [ ] Código ejecutable, modular y documentado.

### 6. Demostración en Vivo (Demo)
- [ ] Demostración funcional en vivo, cronometrada (7-8 minutos).
- [ ] Flujo completo visible: Ingesta/Consulta $\rightarrow$ Búsqueda vectorial $\rightarrow$ Generación con citas de contexto.

---

## ⏱️ Estructura de la Presentación Final (20 - 25 min)

| Segmento | Tiempo | Responsable sugerido |
| :--- | :--- | :--- |
| **Problema y contexto** (necesidad real que se resuelve) | 2 min | |
| **Arquitectura y decisiones de diseño** (C4 + trade-offs) | 5 min | |
| **Ciclo de vida aplicado** (calidad ISO 25010 y seguridad CIA/OWASP) | 6 min | |
| **Demo en vivo del prototipo** (obligatoria y funcional) | 7-8 min | |
| **Lecciones aprendidas y cierre** | 2 min | |
| **Preguntas individuales al equipo** | 2 min | *Todo el equipo* |

---

## 💯 Rúbrica de Evaluación (Nivel "Alcanzado" = 20 pts c/u)

| Criterio | Peso | Requisito para obtener puntaje máximo (Alcanzado) |
| :--- | :---: | :--- |
| **Trabajo colaborativo** | 20% | Coordinación y reparto equitativo evidenciado en el Kanban. Todos participan y cualquiera puede defender cualquier parte del proyecto. |
| **Integración de conocimiento** | 20% | Conecta coherentemente requisitos, arquitectura, calidad y seguridad sobre el caso. RAG se ve aplicado funcionalmente, no solo expuesto. |
| **Investigación** | 20% | Fuentes confiables y actuales que sustentan las decisiones; citadas formalmente; uso de IA declarado, comprendido y defendible. |
| **Diseño integrado** | 20% | Diagrama C4 completo, coherente y justificado por trade-offs. Prototipo funcional (*vertical slice*) alineado a los requisitos. |
| **Ejecución de ingeniería** | 20% | Aplica el ciclo de vida de forma estructurada; demo en vivo sin fallos que evidencia el funcionamiento de punta a punta. |

---

## 📝 Declaración de Uso de IA y Fuentes
> **Nota de cátedra:** El uso de IA está permitido y esperado, pero debe declararse, comprenderse en su totalidad y poder defenderse de forma individual sin depender de la IA.

- **Herramientas de IA utilizadas:**
  - 
- **Propósito del uso:**
  - 
- **Fuentes académicas y técnicas consultadas:**
  -