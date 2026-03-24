# CaseMatch Agent

CaseMatch Agent is an open-source prototype for Chinese criminal similar-case retrieval.

It combines:

- LLM-based structured query extraction
- multi-turn clarification when user information is insufficient
- LanceDB vector recall with SQLite fallback
- BM25 / BGE-M3 / Hybrid reranking
- incremental case import from raw judgment text

The current default corpus is based on [`LeCaRD`](https://github.com/myx666/LeCaRD), and the current schema is criminal-only.

## Features

- Query understanding is not pure text matching. The system extracts structured fields such as `case_summary`, `charges`, `dispute_focus`, `legal_basis`, and `four_elements`.
- Retrieval is two-stage. Candidate cases are recalled from the database first, then reranked by BM25, BGE-M3, or a hybrid scorer.
- The agent always returns a first-round retrieval result before deciding whether it should ask follow-up questions.
- New cases can be imported from raw jsonl through either a CLI script or the Gradio web UI.

## Installation

```bash
pip install -e .
pip install python-dotenv gradio jieba lancedb huggingface_hub
pip install FlagEmbedding
```

## Configuration

```bash
cp .env.example .env
```

All runtime configuration is managed through `.env`. The most important variables are:

- `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL`
- `CASEMATCH_BGE_MODEL_PATH`
- `CASEMATCH_RANKER`
- `CASEMATCH_GRADIO_HOST`, `CASEMATCH_GRADIO_PORT`
- `CASEMATCH_HF_REPO_ID`, `CASEMATCH_HF_REPO_TYPE`, `CASEMATCH_HF_REVISION`, `CASEMATCH_HF_LOCAL_DIR`

Project entrypoints already load `.env` automatically.

## Quick Start

1. Install dependencies.
2. Copy `.env.example` to `.env` and fill your LLM configuration.
3. Download the dataset from Hugging Face.
4. Run the CLI agent or the Gradio UI.

## Download Data from Hugging Face

The public dataset used by this project is:

- `Yuel-P/CaseMatch-Agent-data`

Then set these variables in `.env`:

```dotenv
CASEMATCH_HF_REPO_ID=Yuel-P/CaseMatch-Agent-data
CASEMATCH_HF_REPO_TYPE=dataset
CASEMATCH_HF_REVISION=main
CASEMATCH_HF_LOCAL_DIR=data
HF_TOKEN=
```

`HF_TOKEN` can be left empty if the dataset repo is public.

Download the dataset:

```bash
python scripts/download_hf_data.py
```

If you want to override the repo id manually:

```bash
python scripts/download_hf_data.py \
  --repo-id Yuel-P/CaseMatch-Agent-data \
  --repo-type dataset \
  --revision main \
  --local-dir data
```

After download, your local repository should contain files such as:

```text
data/
  README.md
  lecard/
    README.md
    corpus_merged.jsonl
    queries.jsonl
    qrels.jsonl
    candidate_pools.jsonl
```

## Run the Agent

### CLI

```bash
PYTHONPATH=src python -m casematch_agent
```

### Gradio

```bash
PYTHONPATH=src python -m casematch_agent.gradio_app --host 127.0.0.1 --port 7860
```

The Gradio UI supports:

- multi-turn similar-case retrieval
- structured query display
- retrieval result inspection
- in-page raw case jsonl import

## Build or Rebuild LanceDB

```bash
python scripts/build_lancedb_index.py \
  --corpus data/lecard/corpus_merged.jsonl \
  --lancedb-uri data/cases.lancedb \
  --bge-model-path /data/BAAI/bge-m3
```

Force rebuild:

```bash
python scripts/build_lancedb_index.py \
  --corpus data/lecard/corpus_merged.jsonl \
  --lancedb-uri data/cases.lancedb \
  --bge-model-path /data/BAAI/bge-m3 \
  --force-rebuild
```

## Import New Cases

You can import new criminal cases from raw jsonl.

Input requirement:

- each line must match the `raw_data` schema
- the minimal required fields are:
  - `case_name`
  - `document_name`
  - `fact_text`
  - `judgment_text`
  - `full_text`

Import via CLI:

```bash
python scripts/add_cases_to_db.py \
  --input data/new_cases_raw.jsonl \
  --corpus data/lecard/corpus_merged.jsonl \
  --lancedb-uri data/cases.lancedb \
  --db-backend auto \
  --bge-model-path /data/BAAI/bge-m3
```

Import flow:

1. Read the raw jsonl file
2. Use the LLM API to extract `structured_data`
3. Generate a new non-conflicting random `case_id`
4. Append the record to `data/lecard/corpus_merged.jsonl`
5. Sync LanceDB, or fall back to SQLite refresh in `auto` mode

The Gradio UI exposes the same import flow from the browser.

## Rerankers

### BM25

Use:

```bash
PYTHONPATH=src python -m casematch_agent --ranker bm25
```

Implementation: [src/casematch_ranker/bm25.py](src/casematch_ranker/bm25.py)

### BGE-M3

Use:

```bash
PYTHONPATH=src python -m casematch_agent --ranker bge_m3 --bge-model-path /data/BAAI/bge-m3
```

Implementation: [src/casematch_ranker/bge_m3.py](src/casematch_ranker/bge_m3.py)

### Hybrid

Use:

```bash
PYTHONPATH=src python -m casematch_agent \
  --ranker hybrid \
  --bge-model-path /data/BAAI/bge-m3 \
  --hybrid-bge-weight 1.0 \
  --hybrid-fe-weight 0.1 \
  --hybrid-lc-weight 0.2
```

Implementation: [src/casematch_ranker/hybrid.py](src/casematch_ranker/hybrid.py)

## Offline Experiments

```bash
python scripts/hybrid_experiment.py \
  --corpus data/lecard/corpus_merged.jsonl \
  --queries data/lecard/queries.jsonl \
  --labels data/lecard/qrels.jsonl \
  --candidate-pools data/lecard/candidate_pools.jsonl \
  --methods bm25,bge_m3,hybrid \
  --bge-model-path /data/BAAI/bge-m3 \
  --hybrid-bge-weight 1.0 \
  --hybrid-fe-weight 0.1 \
  --hybrid-lc-weight 0.2
```

For a lighter smoke test:

```bash
python scripts/hybrid_experiment.py \
  --corpus data/lecard/corpus_merged.jsonl \
  --queries data/lecard/queries.jsonl \
  --labels data/lecard/qrels.jsonl \
  --candidate-pools data/lecard/candidate_pools.jsonl \
  --methods bm25 \
  --max-queries 5
```

## Testing

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Project Scope

- Current default corpus: [`LeCaRD`](https://github.com/myx666/LeCaRD)
- Current default schema: criminal-only
- Current primary database: LanceDB
- Current fallback database: SQLite

Schema details are documented in [SCHEMA.md](SCHEMA.md).

## Repository Layout

```text
src/casematch_agent/
  agent.py
  clarification.py
  corpus.py
  extractor.py
  gradio_app.py
  lancedb_store.py
  llm.py
  models.py
  retriever.py
  sqlite_store.py

src/casematch_ranker/
  bm25.py
  bge_m3.py
  hybrid.py

scripts/
  add_cases_to_db.py
  build_lancedb_index.py
  download_hf_data.py
  hybrid_experiment.py
```

## Notes

- If you publish this repository, keep `.env` as a template only. Do not commit real secrets.
- If you use a provider other than OpenAI, make sure it supports the OpenAI-compatible `chat/completions` API shape used in [src/casematch_agent/llm.py](src/casematch_agent/llm.py).

## Acknowledgements

This project benefited from substantial assistance from multiple general-purpose AI systems during design and implementation, especially ChatGPT, Claude, and DeepSeek. They helped with feature design, code iteration, debugging, documentation, and engineering tradeoff exploration throughout the project.
