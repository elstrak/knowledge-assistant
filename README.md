# Knowledge Assistant


### Features\n
- Collects all `.md` notes from an Obsidian vault
- Parses YAML frontmatter, headings, links `[[...]]`, and inline `#tags`
- Produces `notes.jsonl` (one JSON object per note)
- Splits notes into chunks by sections → `chunks.jsonl`


### Project structure\n
- `scripts/collect_obsidian.py`: scan vault → `data/processed/notes.jsonl`
- `scripts/preprocess_obsidian.py`: notes → sections → sentence chunks → `data/processed/chunks.jsonl`
- `data/processed/notes.jsonl`: collected notes
- `data/processed/chunks.jsonl`: generated chunks


### Requirements
- Python 3.9+
- Packages: `PyYAML`

Install locally (recommended virtual env):

```
python -m venv .venv
source .venv/bin/activate 
pip install pyyaml
```

### Quickstart

1) Collect notes from Obsidian vault → `notes.jsonl`

```
python scripts/collect_obsidian.py \
  --vault-path "Obsidian/Obsidian Vault" \
  --output "data/processed/notes.jsonl" \
  --exclude-dir ".obsidian" \
  --exclude-tag "private"
```

2) Generate chunks → `chunks.jsonl`

```
python scripts/preprocess_obsidian.py \
  --input "data/processed/notes.jsonl" \
  --output "data/processed/chunks.jsonl" \
  --chunk-size 300 
  --overlap 60
```

### Data schemas (JSONL)
Each line is a single JSON object.

- Notes (`notes.jsonl`):
  - `id` (str): vault-relative path, e.g. `dl/RNN.md`
  - `title` (str): derived from frontmatter title or first `#` heading (fallback filename)
  - `tags` (list[str]): from frontmatter and inline `#tags`
  - `links` (list[str]): wiki-links targets from `[[Link|Alias]]`
  - `content` (str): markdown body without frontmatter
  - `created` (str|null): ISO timestamp from file stat
  - `modified` (str|null): ISO timestamp from file stat

- Chunks (`chunks.jsonl`):
  - `chunk_id` (str): `<note_id>#<n>`
  - `note_id` (str)
  - `title` (str)
  - `section` (str): section title (based on headings) or note title
  - `text` (str): chunk text (sentences), cleaned of markdown formatting
  - `tags` (list[str])
  - `links` (list[str])
  - `position` (int): 1-based index within the note

