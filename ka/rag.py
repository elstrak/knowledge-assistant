from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ka.embeddings import Embedder, EmbeddingConfig
from ka.jsonl import read_jsonl
from ka.vector_index import IndexConfig, create_best_index, load_best_index


@dataclass(frozen=True)
class RetrievalHit:
    score: float
    chunk_id: str
    note_id: str
    title: str
    section: str
    text: str


def build_index(
    chunks_path: str,
    out_dir: str,
    embed_cfg: Optional[EmbeddingConfig] = None,
    index_cfg: Optional[IndexConfig] = None,
    max_chunks: Optional[int] = None,
) -> None:
    embed_cfg = embed_cfg or EmbeddingConfig()
    index_cfg = index_cfg or IndexConfig()

    rows: List[Dict[str, Any]] = []
    passages: List[str] = []

    for i, row in enumerate(read_jsonl(chunks_path)):
        if max_chunks is not None and i >= max_chunks:
            break
        # Укрепляем сигнал: эмбеддим не только текст чанка, но и мета.
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


class Retriever:
    def __init__(self, index_dir: str, embed_cfg: Optional[EmbeddingConfig] = None):
        idx, embed_model = load_best_index(index_dir)
        self._index = idx
        self._embed_cfg = embed_cfg or EmbeddingConfig(model_name=embed_model or EmbeddingConfig().model_name)
        self._embedder = Embedder(self._embed_cfg)

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalHit]:
        qv = self._embedder.embed_queries([query])[0]
        # Retrieve a bit more and re-rank lexically to make baseline more stable.
        raw = self._index.search(qv, k=max(k * 5, k))
        hits = _lexical_rerank(query, raw)[:k]
        out: List[RetrievalHit] = []
        for score, p in hits:
            out.append(
                RetrievalHit(
                    score=score,
                    chunk_id=str(p.get("chunk_id", "")),
                    note_id=str(p.get("note_id", "")),
                    title=str(p.get("title", "")),
                    section=str(p.get("section", "")),
                    text=str(p.get("text", "")),
                )
            )
        return out


def _lexical_rerank(query: str, hits: List[tuple[float, Dict[str, Any]]]) -> List[tuple[float, Dict[str, Any]]]:
    q_tokens = set(_tokens(query))
    if not q_tokens:
        return hits

    rescored: List[tuple[float, Dict[str, Any]]] = []
    for score, p in hits:
        doc = " ".join(
            [
                str(p.get("title", "") or ""),
                str(p.get("section", "") or ""),
                " ".join([f"#{t}" for t in (p.get("tags", []) or [])]),
                str(p.get("note_id", "") or ""),
                str(p.get("text", "") or ""),
            ]
        )
        d_tokens = set(_tokens(doc))
        overlap = len(q_tokens & d_tokens)
        # small boost: keep vector score primary, but reward lexical match
        boosted = float(score) + 0.03 * float(overlap)
        rescored.append((boosted, p))

    rescored.sort(key=lambda x: x[0], reverse=True)
    return rescored


def _tokens(text: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in ("_", "-"):
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def format_answer_extractively(question: str, hits: List[RetrievalHit]) -> str:
    """
    Baseline "генерация": сшиваем top-k и показываем источники.
    Это честный baseline без LLM, но end-to-end запрос→вывод уже работает.
    """
    if not hits:
        return "Не нашёл релевантных фрагментов в базе."

    lines: List[str] = []
    # ASCII headers to avoid Windows console encoding issues.
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Answer (baseline, retrieval-only):")
    lines.append("")

    for i, h in enumerate(hits, start=1):
        lines.append(f"{i}) [{h.title} -> {h.section}] (score={h.score:.3f})")
        lines.append(h.text)
        lines.append(f"   source: {h.note_id} ({h.chunk_id})")
        lines.append("")

    return "\n".join(lines).strip()


