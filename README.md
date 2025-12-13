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


### Baseline evaluation

1) Create a validation set (weak supervision from `chunks.jsonl`)

```
python scripts/make_validation_set.py --chunks "dataset/processed/chunks.jsonl" --out "dataset/validation/validation.jsonl" --n 100
```

2) Compute retrieval metrics

```
python scripts/evaluate.py --index "dataset/index" --val "dataset/validation/validation.jsonl" --k 5
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

