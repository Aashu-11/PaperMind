## PaperMind: Research-Grade Academic Q&A Engine

PaperMind is an academic question-answering and exploration system that combines **multi-source literature retrieval**, a **vector-based knowledge base**, **LLM synthesis**, and an optional **Neo4j citation graph**. The user interacts through a **Streamlit** UI, while the backend orchestrates calls to arXiv, Semantic Scholar, CrossRef, and OpenAlex, incrementally building a persistent scholarly knowledge graph.

---

## High-Level Overview

- **Goal**: Provide reliable, well-contextualized answers to research questions, grounded in up-to-date academic literature.
- **Frontend**: Streamlit app (`app.py`) offering:
  - Chat-style Q&A interface.
  - Direct database search (raw results).
  - Interactive citation graph visualization (D3.js, backed by Neo4j).
- **Backend Services**:
  - `AcademicSearchEngine` (`src/academic_search.py`) for unified multi-API retrieval.
  - `KnowledgeBase` (`src/knowledge_base.py`) for persistent semantic search using ChromaDB + ONNX embeddings.
  - `QAEngine` (`src/qa_engine.py`) for retrieval-augmented generation (RAG) over multiple LLM providers.
  - `Neo4jClient` (`src/neo4j_client.py`) for storing and querying citation graphs.
- **Configuration**: `Config` (`src/config.py`) reads `.env` and centralizes paths, API keys, and model configuration.

---

## Architecture Diagram

The following diagram reflects the actual code-level architecture of this repository:

```mermaid
flowchart LR
    subgraph UI["Streamlit UI (app.py)"]
        A[User\nQuestions & Searches]
        CH[Chat History (session_state)]
        T1[Tab: Q&A Chat]
        T2[Tab: Direct Search]
        T3[Tab: Citations Graph]
    end

    subgraph Core["Backend Orchestration"]
        CFG[Config\nsrc/config.py]
        ASE[AcademicSearchEngine\nsrc/academic_search.py]
        KB[KnowledgeBase\nChromaDB\nsrc/knowledge_base.py]
        QA[QAEngine\nRAG + LLMs\nsrc/qa_engine.py]
        NEO[Neo4jClient\nCitation Graph\nsrc/neo4j_client.py]
    end

    subgraph LLMs["LLM Providers"]
        L1[Groq\n(llama-3.1-8b-instant)]
        L2[OpenAI\n(gpt-4o)]
        L3[Anthropic\n(claude-3.5-sonnet-20241022)]
        L4[Google\n(gemini-pro)]
    end

    subgraph Retrieval["External Academic APIs"]
        R1[arXiv]
        R2[Semantic Scholar]
        R3[CrossRef]
        R4[OpenAlex]
    end

    subgraph Storage["Persistent Storage"]
        S1[ChromaDB\nVector Store\nsrc/knowledge_base/]
        S2[Neo4j DB\nCitation Graph]
        FS[Local FS\ncache/, .env]
    end

    %% UI interactions
    A --> T1
    A --> T2
    T1 -->|question| QA
    T2 -->|query| ASE

    %% Config wiring
    UI --> CFG
    CFG --> ASE
    CFG --> KB
    CFG --> QA
    CFG --> NEO

    %% Retrieval flows
    ASE -->|search_all()| R1
    ASE -->|search_all()| R2
    ASE -->|search_all()| R3
    ASE -->|search_all()| R4

    %% Knowledge base flows
    QA -->|kb.search()| KB
    ASE -->|results| KB
    KB -->|embeddings & vectors| S1

    %% LLM flows
    QA -->|invoke()| L1
    QA -->|invoke()| L2
    QA -->|invoke()| L3
    QA -->|invoke()| L4

    %% Neo4j graph flows
    QA -->|sources| NEO
    ASE -->|flat_results| NEO
    NEO -->|Query & Paper nodes| S2
    T3 -->|fetch_query_graph()| NEO

    %% Outputs
    QA -->|answer + sources| T1
    ASE -->|flat results| T2
```

This diagram is consistent with the actual module boundaries and method calls in the codebase.

---

## Component-Level Architecture

### 1. Configuration Layer (`src/config.py`)

- **Class**: `Config`
- **Responsibilities**:
  - Loads environment variables via `python-dotenv`.
  - Defines filesystem paths:
    - `BASE_DIR`: root of `src/`.
    - `CACHE_DIR`: location for cached data (currently reserved for extensions).
    - `CHROMA_DIR`: persistent directory for ChromaDB.
  - Centralizes API keys:
    - `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`.
    - `SEMANTIC_SCHOLAR_API_KEY`, `CROSSREF_EMAIL`.
    - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`.
  - Encodes model and provider-level configuration:
    - `DEFAULT_MODEL` (e.g., `"groq"`).
    - `MODELS` mapping provider → underlying model name, temperature, and max tokens.
  - Search and caching configuration:
    - `PAPERS_PER_SOURCE`, `MAX_TOTAL_PAPERS`.
    - `CACHE_RESULTS`, `CACHE_EXPIRY_HOURS` (available for further optimization).
  - Ensures `CACHE_DIR` and `CHROMA_DIR` exist on startup.

**Design implications**:
- All other components depend on `Config` for behavior and credentials.
- Changing `DEFAULT_MODEL` or API keys is entirely declarative via `.env`.

---

### 2. Frontend / Orchestration Layer (`app.py`)

The `Streamlit` app (`app.py`) is responsible for session management, user interaction, and wiring of the main backend objects.

- **Session initialization (`initialize_session`)**:
  - Instantiates:
    - `AcademicSearchEngine(config)`.
    - `KnowledgeBase(str(config.CHROMA_DIR))`.
    - Optional `Neo4jClient` if all `NEO4J_*` variables are present.
    - `QAEngine(config, kb, search_engine)` using the currently selected `DEFAULT_MODEL`.
  - Maintains:
    - `chat_history`, `last_graph_query_id`, `last_graph_sources`, `last_graph_query`.

- **API key validation (`validate_api_keys`)**:
  - Checks for at least one configured LLM provider key.
  - Fails fast in UI if no keys are found, preventing partial initialization.

- **Sidebar (`render_sidebar`)**:
  - Dynamic model selection:
    - Allows user to switch `groq` / `openai` / `anthropic` / `google`.
    - On change, re-instantiates `QAEngine` with updated `DEFAULT_MODEL`.
  - Knowledge base metrics:
    - Displays `Papers Indexed` via `KnowledgeBase.get_stats()`.
    - Provides a button to clear the Chroma collection.
  - API status panel:
    - Shows whether key providers are configured.
  - Embedded high-level explanation of the interaction flow.

- **Q&A Chat (`render_chat`)**:
  - Maintains `st.session_state.chat_history`.
  - On new question:
    - Calls `QAEngine.answer(prompt, use_kb=True, search_new=True)`.
    - Persists the result in chat history.
    - Triggers `record_graph()` to write Neo4j graph for the current query.
  - Presents:
    - Answer text.
    - An expandable section with up to 10 supporting source cards.
    - Simple metrics (total sources, KB vs new sources).

- **Direct Search (`render_direct_search`)**:
  - Calls `AcademicSearchEngine.search_all()` and `flatten_results()`.
  - Shows:
    - Result count.
    - Abstracts and metadata per paper.
    - One-click ingestion of all results into `KnowledgeBase`.
  - Also writes citation graph via `record_graph()`, reusing the same Neo4j schema.

- **Citation Graph (`render_citation_graph`)**:
  - Retrieves the last query’s graph from Neo4j by `query_id`.
  - Renders it using D3.js inside a custom HTML component (`components.html`).
  - Graph nodes correspond to returned papers; edges represent `CITES` relationships inferred from OpenAlex references.

**Design implications**:
- `app.py` remains thin and declarative; most logic is delegated to dedicated services.
- All state persists across interactions via Streamlit `session_state`.

---

### 3. Academic Retrieval Layer (`src/academic_search.py`)

**Class**: `AcademicSearchEngine`

- **Purpose**: Provide a unified abstraction over multiple heterogeneous academic APIs.
- **Entry points**:
  - `search_all(query: str, papers_per_source: int) -> Dict[str, List[Dict]]`:
    - Orchestrates:
      - `_search_arxiv`
      - `_search_semantic_scholar`
      - `_search_crossref`
      - `_search_openalex`
    - Returns a mapping of source name → list of normalized paper dicts.
  - `flatten_results(results: Dict[str, List[Dict]]) -> List[Dict]`:
    - Concatenates and deduplicates papers by normalized title.
    - Prioritizes papers by citation impact and recency.

#### Per-API Responsibilities

- **arXiv (`_search_arxiv`)**:
  - Uses `arxiv.Client` and `arxiv.Search`.
  - Sorts by relevance.
  - Normalized fields:
    - `id`, `title`, `authors`, `abstract`, `published`, `url`, `source`, `citation`, `year`.

- **Semantic Scholar (`_search_semantic_scholar`)**:
  - Hits `https://api.semanticscholar.org/graph/v1/paper/search`.
  - Optional API key via `SEMANTIC_SCHOLAR_API_KEY`.
  - Retries up to 3 times for transient failures.
  - Normalized fields:
    - `paperId`, `title`, `abstract`, `year`, `citationCount`, `fieldsOfStudy`, etc.

- **CrossRef (`_search_crossref`)**:
  - Hits `https://api.crossref.org/works`.
  - Sets a proper `User-Agent` (`PaperMind/2.0`) and `mailto` from `CROSSREF_EMAIL`, which is required by CrossRef TOS.
  - Extracts DOI, journal, authors, and published year.

- **OpenAlex (`_search_openalex`)**:
  - Hits `https://api.openalex.org/works`.
  - Retrieves:
    - `id`, `title`, `authorships`, `abstract`, `publication_year`, `cited_by_count`, `referenced_works`.
  - This is the primary source of citation graph structure via `referenced_works`.

**Design implications**:
- All API differences (schemas, optional fields) are normalized into a consistent internal representation the rest of the system can consume.
- Failures in one provider do not crash the pipeline; each search method is individually exception-safe.

---

### 4. Knowledge Base Layer (`src/knowledge_base.py`)

**Class**: `KnowledgeBase`

- **Backend**: ChromaDB `PersistentClient` with ONNX-based MiniLM embeddings (`ONNXMiniLM_L6_V2`).
- **Collection**: `academic_papers`, with metadata describing the corpus.

#### Key Methods

- `add_papers(papers: List[Dict]) -> int`:
  - For each paper, constructs a compact document text combining:
    - `Title`, `Authors`, `Abstract`, `Year`, `Source`.
  - Builds metadata:
    - `title`, `source`, `citation`, `year`, `paper_id`.
  - Computes a stable Chroma `id` as `md5(title)` to deduplicate across sessions and sources.
  - Adds to the Chroma collection with an embedding function attached to the collection.

- `search(query: str, n_results: int = 10) -> List[Dict]`:
  - Uses `collection.query(query_texts=[query], n_results=n_results)`.
  - Wraps Chroma’s results into a uniform academic paper format, preserving:
    - Content snippet (`content`), `title`, `source`, `citation`, `year`, and optionally `distance`.

- `clear()`:
  - Deletes the entire `academic_papers` collection and recreates it empty.

- `get_stats() -> Dict`:
  - Returns:
    - `total_papers`: `collection.count()`.
    - `collection_name`.

**Design implications**:
- The knowledge base is **persistent** across sessions via `CHROMA_DIR`, enabling longitudinal accumulation of literature.
- RAG can work even without live API calls if the KB is pre-populated.

---

### 5. Question Answering / RAG Layer (`src/qa_engine.py`)

**Class**: `QAEngine`

- **Inputs**:
  - `config`: to determine which LLM provider/model to use.
  - `knowledge_base`: to perform dense retrieval.
  - `search_engine`: to perform on-demand multi-source retrieval.

#### Model Instantiation (`_create_llm`)

- Resolves `provider = config.DEFAULT_MODEL` and `model_config = config.MODELS[provider]`.
- Provider-specific strategies:
  - **Groq**:
    - Preferred: `ChatGroq` from `langchain_groq`.
    - If `ValidationError` or `TypeError` occurs, falls back to a custom `GroqSDKWrapper` that mimics LangChain’s `.invoke()`.
  - **OpenAI**:
    - `ChatOpenAI` with `api_key=config.OPENAI_API_KEY`.
  - **Anthropic**:
    - `ChatAnthropic` with `api_key=config.ANTHROPIC_API_KEY`.
  - **Google**:
    - `ChatGoogleGenerativeAI` with `google_api_key=config.GOOGLE_API_KEY`.

#### Answer Generation (`answer`)

Pipeline:

1. **Initialize source set**.
2. **Retrieve from knowledge base** (if `use_kb`):
   - `kb.search(question, n_results=5)` → KB hits.
3. **Live retrieval** (if `search_new`):
   - `search_engine.search_all(question, papers_per_source=3)` → multi-API results.
   - `flatten_results()` → deduplicated list (top ~10).
   - Ingest new papers into KB via `kb.add_papers(new_papers)` for future reuse.
   - Convert each new paper into a generic `source` dict containing:
     - `content` (title + abstract), `title`, `source`, `citation`, `year`, `openalex_id`, `references`, `citation_count`, `cited_by`.
4. **Context construction**:
   - Select up to 15 sources and serialize them into a textual context via `_format_context`.
5. **LLM prompt**:
   - Builds a structured system-style prompt instructing the model to:
     - Base the answer strictly on supplied papers.
     - Provide a natural ChatGPT-style explanation.
     - Summarize consensus and highlight nuances.
6. **LLM invocation**:
   - `self.llm.invoke(prompt)` (LangChain or `GroqSDKWrapper` interface).
7. **Packaging response**:
   - Returns:
     - `answer` text.
     - `sources` (combined KB and live).
     - `total_sources`, `kb_sources`, and `new_sources` counts.

In case of LLM failure, a meaningful error message is returned but the sources are preserved for debugging or fallback behavior.

#### GroqSDKWrapper

- Wraps the official `groq` Python SDK.
- Exposes an `.invoke(prompt)` method returning a simple object with `.content`.

**Design implications**:
- The RAG logic is modular: changing retrieval or LLM providers does not require GUI changes.
- The answer object carries full provenance (sources, counts), enabling advanced visualization or evaluation later.

---

### 6. Citation Graph Layer (`src/neo4j_client.py`)

**Class**: `Neo4jClient`

- **Purpose**: Persist a **query-centric citation subgraph** for interactive visualization and downstream analysis.
- **Backend**: Neo4j (via official `neo4j` Python driver).

#### Graph Schema

- **Nodes**:
  - `(:Query {id, text, updated_at})`
  - `(:Paper {id, title, source, year, citation, cited_by})`
- **Relationships**:
  - `(:Query)-[:RETURNED]->(:Paper)`
  - `(:Paper)-[:CITES]->(:Paper)`

#### Ingestion (`upsert_query_graph`)

1. Computes `query_id = md5(normalized_query)`.
2. For each paper:
   - Generates a stable internal ID via `_paper_id` (OpenAlex ID if present, else source+id, else citation, else md5(title)).
   - Deduplicates papers within the current batch.
3. Writes:
   - `MERGE` on `Query` node with `id=query_id`.
   - `MERGE` on `Paper` nodes with attributes including `cited_by` (citation count).
   - Connects `Query` to `Paper` with `[:RETURNED]` edges.
4. Constructs citation edges:
   - Uses `openalex_id` and `references` fields from OpenAlex-based results.
   - Adds `(:Paper)-[:CITES]->(:Paper)` edges between any pair of returned papers where one references the other.

#### Query (`fetch_query_graph`)

- Given a `query_id`, it:
  - Retrieves all `Paper` nodes connected to that `Query`.
  - Extracts citation edges among those papers.
  - Returns a JSON-serializable graph:
    - `nodes`: with `id`, `label`, `group`, `size` (based on `cited_by`), and `meta`.
    - `links`: list of `{source, target}` edges.
- This structure is directly consumed by the D3.js force-directed graph in `app.py`.

**Design implications**:
- The citation graph is compact and query-scoped, optimized for interactive visualization.
- The schema is simple but extensible (e.g., additional properties or relationship types can be appended).

---

## Data Flow Summary

1. **User question** (Q&A tab):
   - UI → `QAEngine.answer`.
   - `KnowledgeBase.search` (optional) + `AcademicSearchEngine.search_all`.
   - New papers ingested into ChromaDB.
   - LLM synthesizes answer from compiled context.
   - Neo4j graph updated for the current query.
   - UI displays answer, metrics, sources, and makes graph available.

2. **Direct search**:
   - UI → `AcademicSearchEngine.search_all`.
   - Results flattened and optional ingestion into KB.
   - Neo4j graph written for the raw search query.
   - UI lists raw results and offers “Add All to Knowledge Base”.

3. **Citation graph visualization**:
   - UI requests `fetch_query_graph(last_graph_query_id)`.
   - Neo4j returns nodes and edges.
   - D3.js renders an interactive force-directed layout.

---

## Environment & Configuration

Create a `.env` file at the project root with the following keys as needed:

```bash
GROQ_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
SEMANTIC_SCHOLAR_API_KEY=...
CROSSREF_EMAIL=your-email@example.com

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

DEFAULT_MODEL=groq   # or openai / anthropic / google
```

- Only one LLM key is required to run, but multiple keys allow dynamic switching from the sidebar.
- Neo4j configuration is optional; if omitted, graph features degrade gracefully in the UI.

---

## Running the System

1. **Install dependencies** (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

2. **Start Neo4j** (optional but recommended for full functionality).

3. **Run the Streamlit app** from the project root:

```bash
streamlit run app.py
```

4. Open the provided local URL in your browser and:
   - Configure model provider and inspect KB stats in the sidebar.
   - Use the **Q&A Chat** tab for research questions.
   - Use **Direct Search** for raw multi-database exploration.
   - Use **Citations Graph** to explore structural relationships.

---

## Research & Extension Directions

This repository is designed for research-level experimentation. Some possible extensions:

- **Retrieval strategies**:
  - Hybrid lexical + dense retrieval (e.g., BM25 + Chroma).
  - Time-aware re-ranking (favoring more recent publications).
- **Evaluation**:
  - Intrinsic metrics: coverage of retrieved sources, redundancy reduction.
  - Extrinsic: expert evaluation of answer quality and citation fidelity.
- **Graph analytics**:
  - Graph-based influence scores (e.g., PageRank over `:CITES` subgraph).
  - Community detection among returned papers.
- **UI/UX improvements**:
  - Source-aware highlighting in answers.
  - Interactive filters for year, venue, field of study.

The current architecture isolates concerns cleanly (retrieval, storage, RAG, graph), making it straightforward to swap out or extend individual layers without disruptive changes to the whole system.

