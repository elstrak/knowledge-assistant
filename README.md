# Knowledge Assistant

Knowledge Assistant is a baseline RAG-style pipeline for Obsidian notes: it collects Markdown notes from a local vault, preprocesses them into chunks, builds a vector index, and answers queries locally (CLI or API) by retrieving the most relevant chunks with sources.


### Features
- Collect `.md` notes from an Obsidian vault into `notes.jsonl`
- Parse YAML frontmatter, wiki-links `[[...]]`, and inline `#tags`
- Chunk notes into `chunks.jsonl` (sections + sentence-based chunking)
- Build a local vector index from chunks
- Query end-to-end via CLI or local FastAPI server
- Baseline evaluation with Recall@k / MRR@k on a validation set


### Project structure
- `scripts/collect_obsidian.py`: scan vault → `notes.jsonl`
- `scripts/preprocess_obsidian.py`: `notes.jsonl` → `chunks.jsonl`
- `scripts/build_index.py`: `chunks.jsonl` → `dataset/index/`
- `scripts/ask.py`: ask questions (agent + retrieval baseline)
- `scripts/make_validation_set.py`: build `dataset/validation/validation.jsonl`
- `scripts/evaluate.py`: compute Recall@k / MRR@k
- `ka/`: core package (embeddings, index, retriever, agent, server)


### Requirements
- Python 3.9+ (baseline works on Windows; tested with Python 3.13)
- Install dependencies from `requirements.txt`

Setup (Windows PowerShell):

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies
The baseline pipeline is designed to run without heavy ML stacks.

Core dependencies (used in baseline):
- `PyYAML`: YAML frontmatter parsing
- `numpy`: baseline embeddings + cosine retrieval
- `tqdm`: progress bars
- `fastapi`, `uvicorn`: local HTTP API
- `requests`: utilities / simple HTTP client

Optional (quality/speed upgrades, not required for baseline):
- `sentence-transformers` (+ `torch`): semantic embeddings
- `hnswlib`: fast ANN vector index (HNSW)

### Environment

This project loads environment variables from a `.env` file in the repo root (via `python-dotenv`). 

Setup:
- Copy example here to .env
- Fill your LLM API key
- Optionally adjust defaults and retrieval tuning

```
KA_LLM_BASE_URL=https://api.mistral.ai
KA_LLM_MODEL=mistral-large-latest
KA_LLM_API_KEY=YOUR_KEY

KA_LLM_TIMEOUT_S=60
KA_CONTEXT_CHARS=12000
KA_MAX_CHUNKS_PER_NOTE=2
KA_RRF_K=60
KA_LLM_MAX_TOKENS=1200
KA_LLM_TEMPERATURE=0.2
```

### Quickstart (end-to-end)

1) Collect notes from an Obsidian vault → `notes.jsonl`

```
python scripts/collect_obsidian.py ^
  --vault-path "Obsidian/Obsidian Vault" ^
  --output "dataset/processed/notes.jsonl" ^
  --exclude-dir ".obsidian" ^
  --exclude-tag "private"
```

2) Chunk notes → `chunks.jsonl`

```
python scripts/preprocess_obsidian.py ^
  --input "dataset/processed/notes.jsonl" ^
  --output "dataset/processed/chunks.jsonl" ^
  --chunk-size 300 ^
  --overlap 60
```

3) Build vector index → `dataset/index/`

```
python scripts/build_index.py ^
  --chunks "dataset/processed/chunks.jsonl" ^
  --out "dataset/index" ^
  --backend hashing
```

4) Ask a question (CLI)

```
python scripts/ask.py --index "dataset/index" --notes "dataset/processed/notes.jsonl" --q "What is the walrus operator in Python?"
```

5) Run local API

```
uvicorn ka.server:app --host 127.0.0.1 --port 8000
```


### Evaluation

1) Use a validation set to compute retrieval metrics

```
python scripts/evaluate_retriever.py --index dataset/index --validation dataset/validation/validation_set.jsonl --k 1,3,5,10
```

2) Use a validation set to evaluate generator

```
python scripts/evaluate_rag.py --index dataset/index --validation dataset/validation/validation_set.jsonl --k 5 --judge --judge_n 20
```


### Data schemas (JSONL)
Each line is a single JSON object.

- Notes (`notes.jsonl`):
  - `id` (str): vault-relative path
  - `title` (str)
  - `tags` (list[str])
  - `links` (list[str])
  - `content` (str)
  - `created` (str|null)
  - `modified` (str|null)

- Chunks (`chunks.jsonl`):
  - `chunk_id` (str): `<note_id>#<n>`
  - `note_id` (str)
  - `title` (str)
  - `section` (str)
  - `text` (str)
  - `tags` (list[str])
  - `links` (list[str])
  - `position` (int)

