import os, re, math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from ka.embeddings import Embedder, EmbeddingConfig
from ka.vector_index import load_best_index

STOPWORDS = {
    # RU
    "и","в","во","на","к","ко","о","об","от","до","из","у","по","за","при","без",
    "что","это","я","мы","ты","вы","он","она","они","оно",
    "как","про","для","с","со","а","но","или","ли","же","бы","быть","есть",
    "не","ни","да","нет","тоже","еще","ещё","уже","тут","там","так",
    "мой","твой","наш","ваш","его","ее","её","их","этот","эта","эти","тот","та","те",
    # EN
    "the","a","an","and","or","to","of","in","on","for","with","without","is","are","was","were",
    "i","you","we","they","he","she","it","this","that","these","those",
}

_ASCII_RE = re.compile(r"[a-z0-9]", re.IGNORECASE)
_TAG_RE = re.compile(r"(?<!\w)#([\w/-]+)", re.UNICODE)

@dataclass(frozen=True)
class RetrievalHit:
    score: float
    chunk_id: str
    note_id: str
    title: str
    section: str
    text: str


class Retriever:
    def __init__(self, index_dir: str, embed_cfg: Optional[EmbeddingConfig] = None):
        idx, embed_model = load_best_index(index_dir)
        self._index = idx
        self._embed_cfg = embed_cfg or EmbeddingConfig(model_name=embed_model or EmbeddingConfig().model_name)
        self._embedder = Embedder(self._embed_cfg)

        # Гибридный ретривер (Vector + BM25) 
        payloads = list(_iter_index_payloads(idx))
        self._bm25 = _BM25(payloads)
        self._alias_to_note_id = _build_note_aliases(payloads)

        # Controls
        self._diversify_by_note = os.getenv("KA_DIVERSIFY_BY_NOTE", "1") != "0"
        self._max_chunks_per_note = int(os.getenv("KA_MAX_CHUNKS_PER_NOTE", "2"))
        self._rrf_k = int(os.getenv("KA_RRF_K", "60"))

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalHit]:
        k = max(1, int(k))
        overfetch = max(k * 8, 40)

        # 1) Vector retrieval
        qv = self._embedder.embed_queries([query])[0]
        vec_hits = self._index.search(qv, k=overfetch)

        # 2) BM25 retrieval
        bm25_hits = self._bm25.search(query, k=overfetch)

        # 3) слияние рангов с prf
        fused = _rrf_fuse(vec_hits, bm25_hits, rrf_k=self._rrf_k)

        # 4) бустим графы и теги
        query_tags = _extract_tags_from_text(query)
        related_notes = _collect_related_notes([p for _, p in vec_hits[: min(10, len(vec_hits))]], self._alias_to_note_id)

        rescored: List[Tuple[float, Dict[str, Any]]] = []
        for score, p in fused:
            s = float(score)
            if query_tags:
                ptags = set(str(t).lower() for t in (p.get("tags", []) or []) if t)
                overlap = len(set(t.lower() for t in query_tags) & ptags)
                if overlap:
                    s += 0.35 * overlap
            if related_notes and str(p.get("note_id", "")) in related_notes:
                s += 0.10
            rescored.append((s, p))
        
        # 5) защита ключевых слов
        if os.getenv("KA_KEYWORD_GUARD", "1") != "0":
            keywords = _extract_query_keywords(query, self._bm25)
        else:
            keywords = []

        if keywords:
            penalty = float(os.getenv("KA_KEYWORD_PENALTY", "1.0"))
            bonus = float(os.getenv("KA_KEYWORD_BONUS", "0.15"))
            for idx, (s, p) in enumerate(rescored):
                ov = _keyword_overlap_count(keywords, p)
                if ov == 0:
                    # если в чанке нет ни одного ключевого слова запроса — сильно вниз
                    s -= penalty
                else:
                    # лёгкий бонус за совпадение
                    s += bonus * ov
                rescored[idx] = (s, p)

        rescored.sort(key=lambda x: x[0], reverse=True)

        # 5) разнообразие заметок (чтобы в top-k не было слишком много чанков из одной заметки)
        if self._diversify_by_note and self._max_chunks_per_note > 0:
            rescored = _diversify_by_note(rescored, max_per_note=self._max_chunks_per_note)

        top = rescored[:k]
        out: List[RetrievalHit] = []
        for score, p in top:
            out.append(
                RetrievalHit(
                    score=float(score),
                    chunk_id=str(p.get("chunk_id", "")),
                    note_id=str(p.get("note_id", "")),
                    title=str(p.get("title", "")),
                    section=str(p.get("section", "")),
                    text=str(p.get("text", "")),
                )
            )
        return out


class _BM25:
    def __init__(self, payloads: List[Dict[str, Any]], k1: float = 1.2, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)

        self.payloads = payloads
        self.N = len(payloads)
        self.doc_len: List[int] = [0] * self.N
        self.avgdl: float = 0.0
        self.inv: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_id, tf), ...]
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

        if self.N == 0:
            return

        total_len = 0
        for doc_id, p in enumerate(payloads):
            text = _payload_text_for_lex(p)
            toks = _tokens(text)
            total_len += len(toks)
            self.doc_len[doc_id] = len(toks)
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for term, cnt in tf.items():
                self.inv.setdefault(term, []).append((doc_id, cnt))
            for term in tf.keys():
                self.df[term] = self.df.get(term, 0) + 1

        self.avgdl = (total_len / self.N) if self.N else 0.0
        for term, df in self.df.items():
            self.idf[term] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 50) -> List[Tuple[float, Dict[str, Any]]]:
        if self.N == 0:
            return []
        q_terms = _tokens(query)
        if not q_terms:
            return []
        q_terms = list(dict.fromkeys(q_terms))  # unique

        scores: Dict[int, float] = {}
        for term in q_terms:
            postings = self.inv.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_id, tf in postings:
                dl = self.doc_len[doc_id] or 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                s = idf * (tf * (self.k1 + 1.0) / denom)
                scores[doc_id] = scores.get(doc_id, 0.0) + s

        if not scores:
            return []
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(1, int(k))]
        return [(float(s), self.payloads[doc_id]) for doc_id, s in top]


def _iter_index_payloads(idx: Any) -> Iterable[Dict[str, Any]]:
    if hasattr(idx, "_payload"):
        payload = getattr(idx, "_payload")
        if isinstance(payload, dict):
            for _, p in payload.items():
                if isinstance(p, dict):
                    yield p
            return
        if isinstance(payload, list):
            for p in payload:
                if isinstance(p, dict):
                    yield p
            return

def _payload_text_for_lex(p: Dict[str, Any]) -> str:
    title = str(p.get("title", "") or "")
    section = str(p.get("section", "") or "")
    note_id = str(p.get("note_id", "") or "")
    tags = " ".join([f"#{t}" for t in (p.get("tags", []) or []) if isinstance(t, str) and t])
    text = str(p.get("text", "") or "")
    return f"{title}\n{section}\n{tags}\n{note_id}\n{text}".strip()


def _rrf_fuse(
    vec_hits: List[Tuple[float, Dict[str, Any]]],
    bm25_hits: List[Tuple[float, Dict[str, Any]]],
    rrf_k: int = 60,
) -> List[Tuple[float, Dict[str, Any]]]:
    rrf_k = max(1, int(rrf_k))
    scores: Dict[str, float] = {}
    best_payload: Dict[str, Dict[str, Any]] = {}

    def add(rank: int, p: Dict[str, Any]) -> None:
        cid = str(p.get("chunk_id", ""))
        if not cid:
            return
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        if cid not in best_payload:
            best_payload[cid] = p

    for i, (_, p) in enumerate(vec_hits, start=1):
        add(i, p)
    for i, (_, p) in enumerate(bm25_hits, start=1):
        add(i, p)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(float(s), best_payload[cid]) for cid, s in fused]


def _extract_tags_from_text(text: str) -> List[str]:
    tags = [m.strip() for m in _TAG_RE.findall(text or "")]
    out: List[str] = []
    seen = set()
    for t in tags:
        low = t.lower()
        if low and low not in seen:
            seen.add(low)
            out.append(t)
    return out

def _build_note_aliases(payloads: List[Dict[str, Any]]) -> Dict[str, str]:
    """alias -> note_id map (best-effort), used to resolve [[wikilinks]]."""
    alias: Dict[str, str] = {}
    for p in payloads:
        note_id = str(p.get("note_id", "") or "")
        if not note_id:
            continue

        alias.setdefault(note_id.lower(), note_id)

        no_ext = note_id[:-3] if note_id.lower().endswith(".md") else note_id
        alias.setdefault(no_ext.lower(), note_id)

        base = no_ext.split("/")[-1]
        if base:
            alias.setdefault(base.lower(), note_id)

        title = str(p.get("title", "") or "").strip()
        if title:
            alias.setdefault(title.lower(), note_id)

    return alias

def _collect_related_notes(
    payloads: List[Dict[str, Any]],
    alias_to_note_id: Dict[str, str],
    max_notes: int = 50,
) -> set[str]:
    out: set[str] = set()
    for p in payloads:
        links = p.get("links", []) or []
        if not isinstance(links, list):
            continue
        for l in links:
            if not isinstance(l, str) or not l.strip():
                continue
            key = l.strip()
            key = key.split("|", 1)[0].split("#", 1)[0].strip()
            note_id = alias_to_note_id.get(key.lower())
            if note_id:
                out.add(note_id)
                if len(out) >= max_notes:
                    return out
    return out

def _diversify_by_note(
    hits: List[Tuple[float, Dict[str, Any]]],
    max_per_note: int = 2,
) -> List[Tuple[float, Dict[str, Any]]]:
    max_per_note = max(1, int(max_per_note))
    counts: Dict[str, int] = {}
    out: List[Tuple[float, Dict[str, Any]]] = []
    for s, p in hits:
        note_id = str(p.get("note_id", "") or "")
        if not note_id:
            out.append((s, p))
            continue
        c = counts.get(note_id, 0)
        if c >= max_per_note:
            continue
        counts[note_id] = c + 1
        out.append((s, p))
    return out


def _extract_query_keywords(query: str, bm25: "_BM25") -> List[str]:
    """
    Берём смысловые токены запроса.
    Логика:
      - токены после стоп-слов
      - оставляем те, что:
          * содержат латиницу/цифры (rag, obsidian, llm, hnsw, faiss)
          * ИЛИ имеют высокий idf в BM25
    """
    toks = _tokens(query)
    if not toks:
        return []
    uniq = list(dict.fromkeys(toks))

    idf_min = float(os.getenv("KA_KEYWORD_IDF_MIN", "1.5"))
    out: List[str] = []
    for t in uniq:
        if _ASCII_RE.search(t):
            out.append(t)
            continue
        if bm25.idf.get(t, 0.0) >= idf_min:
            out.append(t)

    return out[: int(os.getenv("KA_KEYWORD_MAX", "8"))]


def _keyword_overlap_count(keywords: List[str], payload: Dict[str, Any]) -> int:
    # считаем по “lex тексту” (title/section/tags/note_id/text)
    doc_toks = set(_tokens(_payload_text_for_lex(payload)))
    return sum(1 for k in keywords if k in doc_toks)


def _tokens(text: str) -> List[str]:
    """
    Tokenize + remove stopwords and very short tokens.
    Keeps latin/cyrillic/digits/_/-.
    """
    out: List[str] = []
    cur: List[str] = []
    for ch in (text or "").lower():
        if ch.isalnum() or ch in ("_", "-"):
            cur.append(ch)
        else:
            if cur:
                tok = "".join(cur)
                cur = []
                if len(tok) >= 2 and tok not in STOPWORDS:
                    out.append(tok)
    if cur:
        tok = "".join(cur)
        if len(tok) >= 2 and tok not in STOPWORDS:
            out.append(tok)
    return out
