# Cosmetic Chemical Disclosure RAG POC

A local-first Streamlit assistant for California Safe Cosmetics Program disclosures. DuckDB is the factual source of truth, embedded Qdrant discovers semantic chemical candidates, and Ollama runs embeddings and answer synthesis. No hosted LLM API or paid service is used.

## Architecture

```text
app/streamlit_app.py
        |
src/graph/workflow.py  LangGraph orchestration and typed shared state
        |
src/agents/             planner, extraction, retrieval, synthesis
        |                 src/guardrails/ validates scope and evidence
        +-----------------------------+
        |                             |
src/tools/structured_query.py   src/retrieval/vector_store.py
        |                             |
DuckDB source of truth          Embedded Qdrant vector index
                                      |
                               Ollama embeddings
```

Exact CAS, chemical, company, brand, product, category, and date filters use parameterized DuckDB queries. Fuzzy chemical questions retrieve vector K=10 and BM25 K=10, fuse ranks with RRF, cap at 10 candidates, rerank by dense relevance, then accept only scores strictly above 0.70. Accepted names are verified with exact DuckDB queries. MMR is implemented and benchmarked but is not the production default because it did not improve the measured ranking metrics. Answer synthesis uses the local Ollama model when enabled and deterministic text otherwise. Record answers preserve CDPHId and ChemicalId citations.

## Repository structure

```text
app/streamlit_app.py
src/config/                 local paths and Ollama model settings
src/schemas/                Pydantic query/evidence contracts and graph state
src/tools/                  DuckDB, semantic-search, and aggregate tools
src/agents/                 planner, extraction, retrieval, synthesis
src/retrieval/              Ollama embeddings and persistent local Qdrant
src/graph/                  LangGraph workflow and router
src/guardrails/             scope, grounding, and output validation
scripts/                    data utilities and vector-index builder
tests/                      query, agent, vector, graph, model, and UI tests
data/raw/                   supplied CSV
data/processed/             DuckDB database
vectorstore/qdrant/          generated index; ignored by Git
```

The root `structured_query.py` remains a compatibility shim; new imports should use `src.tools.structured_query`.

## Local models and setup

The default models are `qwen2.5:1.5b` for synthesis and `nomic-embed-text` for embeddings. Both run locally through Ollama. They are installed on the development machine but are not bundled in this repository.

Install project packages in Python 3.11+:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install Ollama for Windows, then fetch models if they are not already present:

```powershell
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

If `ollama` is not on the current PowerShell PATH after installation, restart the shell or invoke `%LOCALAPPDATA%\Programs\Ollama\ollama.exe` directly. Copy `.env.example` to `.env` to override local settings. The UI uses no remote fonts or API calls.

Build or rebuild the vector index:

```powershell
.\.venv\Scripts\python.exe scripts\build_vector_index.py
.\.venv\Scripts\python.exe scripts\build_vector_index.py --force
```

Start the demo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

The sidebar reports local model status and can disable LLM synthesis. Results include intent, entities, query trace, per-step latency/work details, local prompt/completion token counts, and $0 hosted API cost. Structured search and deterministic answers do not require the answer model; when vector search is unavailable, BM25/fuzzy candidates are shown but do not bypass the strict relevance cutoff.

## Tests and data tools

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\inspect_dataset.py
.\.venv\Scripts\python.exe scripts\test_queries.py
```

Vector and local-model tests skip when Ollama or the configured models are unavailable. Rebuilding the database with `scripts\create_database.py` replaces the existing `cosmetics` table.

Run the labeled K=5/10/20 retrieval benchmark and write the report to `reports/retrieval_benchmark.md`:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_retrieval --llm-judge
```

The benchmark reports Precision@K, Recall@K, HitRate@K, Accuracy@1, MRR, MAP, NDCG, retrieval latency, MMR results, and independent local LLM-judge agreement. Curated labels are the primary metric truth; the local judge is diagnostic only. The current nine-query sample favors K=10 over K=20 on efficiency with tied unthresholded RRF NDCG/MRR. The strict 0.70 relevance cutoff reduces recall on typo/synonym queries, so that precision-versus-recall tradeoff is explicit in the report.

## Dataset limits

The checked-in database has 114,635 ingredient records across 36,972 products. `MostRecentDateReported` currently ends in 2020; there are no records discontinued in 2024. Zero results describe this dataset only and do not prove that no such products exist elsewhere. The disclosure records do not establish product safety, exposure, or individual health risk.