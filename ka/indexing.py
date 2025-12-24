from __future__ import annotations

from typing import Any, Dict, List, Optional

from ka.embeddings import Embedder, EmbeddingConfig
from ka.jsonl import read_jsonl
from ka.vector_index import create_best_index


def build_index(
    chunks_path: str,
    out_dir: str,
    embed_cfg: Optional[EmbeddingConfig] = None,
    max_chunks: Optional[int] = None,
) -> None:
    embed_cfg = embed_cfg or EmbeddingConfig()

    rows: List[Dict[str, Any]] = []
    passages: List[str] = []

    for i, row in enumerate(read_jsonl(chunks_path)):
        if max_chunks is not None and i >= max_chunks:
            break

        title = row.get("title", "") or ""
        section = row.get("section", "") or ""
        text = row.get("text", "") or ""
        tags = row.get("tags", []) or []
        tags_text = " ".join([f"#{t}" for t in tags if isinstance(t, str) and t])
        note_id = row.get("note_id", "") or ""
        passage = f"{title}\n{section}\n{tags_text}\n{note_id}\n{text}".strip()

        passages.append(passage)
        rows.append(row)

    if not rows:
        raise SystemExit(f"Пустой chunks.jsonl: {chunks_path}")

    embedder = Embedder(embed_cfg)
    vecs = embedder.embed_passages(passages)

    idx = create_best_index(dim=int(vecs.shape[1]), prefer_hnsw=True)

    payloads: List[Dict[str, Any]] = []
    for row in rows:
        payloads.append(
            {
                "chunk_id": row.get("chunk_id"),
                "note_id": row.get("note_id"),
                "title": row.get("title"),
                "section": row.get("section"),
                "text": row.get("text"),
                "tags": row.get("tags", []),
                "links": row.get("links", []),
                "position": row.get("position"),
            }
        )

    idx.add(vecs, payloads)

    if hasattr(idx, "set_query_ef"):
        try:
            idx.set_query_ef(128)  # type: ignore[attr-defined]
        except Exception:
            pass

    idx.save(out_dir=out_dir, embed_model=embed_cfg.model_name)  # type: ignore[arg-type]
