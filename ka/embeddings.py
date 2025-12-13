from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "intfloat/multilingual-e5-small"  # used for sentence-transformers backend
    normalize: bool = True
    backend: str = "auto"  # auto | sentence-transformers | hashing
    hashing_dim: int = 4096


class _Backend(Protocol):
    def embed_queries(self, queries: Iterable[str]) -> np.ndarray: ...
    def embed_passages(self, passages: Iterable[str]) -> np.ndarray: ...


class Embedder:
    """
    Embeddings wrapper with a safe default for Windows + Python 3.13:
    - backend=auto: tries sentence-transformers, falls back to hashing embeddings (pure numpy)
    - backend=hashing: always use hashing
    - backend=sentence-transformers: require sentence-transformers (and torch)
    """

    def __init__(self, cfg: Optional[EmbeddingConfig] = None):
        self.cfg = cfg or EmbeddingConfig()
        self._impl: _Backend = self._init_backend()

    def _init_backend(self) -> _Backend:
        b = (self.cfg.backend or "auto").lower()
        if b == "hashing":
            return _HashingBackend(dim=self.cfg.hashing_dim, normalize=self.cfg.normalize)
        if b == "sentence-transformers":
            return _SentenceTransformersBackend(model_name=self.cfg.model_name, normalize=self.cfg.normalize)
        # auto
        try:
            return _SentenceTransformersBackend(model_name=self.cfg.model_name, normalize=self.cfg.normalize)
        except Exception:
            return _HashingBackend(dim=self.cfg.hashing_dim, normalize=self.cfg.normalize)

    def embed_queries(self, queries: Iterable[str]) -> np.ndarray:
        return self._impl.embed_queries(queries)

    def embed_passages(self, passages: Iterable[str]) -> np.ndarray:
        return self._impl.embed_passages(passages)


class _SentenceTransformersBackend:
    def __init__(self, model_name: str, normalize: bool):
        self.model_name = model_name
        self.normalize = normalize
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(self.model_name)

    def _maybe_prefix(self, texts: List[str], kind: str) -> List[str]:
        if "e5" in self.model_name.lower():
            pref = "query:" if kind == "query" else "passage:"
            return [f"{pref} {t}" for t in texts]
        return texts

    def embed_queries(self, queries: Iterable[str]) -> np.ndarray:
        qs = self._maybe_prefix(list(queries), "query")
        vec = self._model.encode(qs, show_progress_bar=False)
        arr = np.asarray(vec, dtype=np.float32)
        return _l2_normalize(arr) if self.normalize else arr

    def embed_passages(self, passages: Iterable[str]) -> np.ndarray:
        ps = self._maybe_prefix(list(passages), "passage")
        vec = self._model.encode(ps, show_progress_bar=True)
        arr = np.asarray(vec, dtype=np.float32)
        return _l2_normalize(arr) if self.normalize else arr


class _HashingBackend:
    def __init__(self, dim: int, normalize: bool):
        self.dim = int(dim)
        self.normalize = normalize

    def embed_queries(self, queries: Iterable[str]) -> np.ndarray:
        return self._embed(list(queries))

    def embed_passages(self, passages: Iterable[str]) -> np.ndarray:
        return self._embed(list(passages))

    def _embed(self, texts: List[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _tokenize(t):
                j = _stable_hash(tok) % self.dim
                mat[i, j] += 1.0
        if self.normalize:
            mat = _l2_normalize(mat)
        return mat


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def _tokenize(text: str) -> List[str]:
    # super-simple tokenizer (ru/en): letters+digits+_-
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


def _stable_hash(token: str) -> int:
    # stable across runs (unlike Python's built-in hash)
    import hashlib

    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


