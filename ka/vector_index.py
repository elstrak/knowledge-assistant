from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class IndexConfig:
    space: str = "cosine"  # cosine works well with normalized embeddings
    ef_construction: int = 200
    M: int = 48


class HnswIndex:
    """
    Thin wrapper around hnswlib that stores:
    - index binary
    - meta.json (cfg + mapping internal_id -> payload)
    """

    def __init__(self, dim: int, cfg: Optional[IndexConfig] = None):
        self.dim = dim
        self.cfg = cfg or IndexConfig()
        try:
            import hnswlib  # type: ignore
        except ImportError as e:
            raise ImportError("hnswlib is not installed") from e

        self._hnswlib = hnswlib
        self._index = self._hnswlib.Index(space=self.cfg.space, dim=self.dim)
        self._payload: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0

    def add(self, vectors: np.ndarray, payloads: List[Dict[str, Any]]) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Bad vectors shape: {vectors.shape}, expected (*, {self.dim})")
        if len(payloads) != vectors.shape[0]:
            raise ValueError("vectors and payloads length mismatch")

        ids = np.arange(self._next_id, self._next_id + vectors.shape[0], dtype=np.int64)
        if self._next_id == 0:
            self._index.init_index(
                max_elements=int(vectors.shape[0]),
                ef_construction=self.cfg.ef_construction,
                M=self.cfg.M,
            )
        else:
            self._index.resize_index(self._next_id + int(vectors.shape[0]))

        self._index.add_items(vectors, ids)
        for i, p in zip(ids.tolist(), payloads):
            self._payload[int(i)] = p
        self._next_id += int(vectors.shape[0])

    def set_query_ef(self, ef: int) -> None:
        self._index.set_ef(ef)

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[float, Dict[str, Any]]]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        labels, distances = self._index.knn_query(query_vec, k=k)
        res: List[Tuple[float, Dict[str, Any]]] = []
        for lab, dist in zip(labels[0].tolist(), distances[0].tolist()):
            if lab == -1:
                continue
            payload = self._payload[int(lab)]
            # hnswlib for cosine returns distance in [0..2], smaller is better; convert to similarity-ish.
            score = 1.0 - float(dist)
            res.append((score, payload))
        return res

    def save(self, out_dir: str, embed_model: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        index_path = os.path.join(out_dir, "hnsw.index")
        meta_path = os.path.join(out_dir, "meta.json")

        self._index.save_index(index_path)
        meta = {
            "dim": self.dim,
            "space": self.cfg.space,
            "ef_construction": self.cfg.ef_construction,
            "M": self.cfg.M,
            "embed_model": embed_model,
            "payload": self._payload,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, out_dir: str) -> Tuple["HnswIndex", str]:
        index_path = os.path.join(out_dir, "hnsw.index")
        meta_path = os.path.join(out_dir, "meta.json")
        if not os.path.isfile(index_path) or not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Index not found in {out_dir}. Expected files: hnsw.index, meta.json"
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        dim = int(meta["dim"])
        cfg = IndexConfig(
            space=str(meta.get("space", "cosine")),
            ef_construction=int(meta.get("ef_construction", 200)),
            M=int(meta.get("M", 48)),
        )
        embed_model = str(meta.get("embed_model", ""))

        idx = cls(dim=dim, cfg=cfg)
        idx._payload = {int(k): v for k, v in meta.get("payload", {}).items()}
        idx._next_id = (max(idx._payload.keys()) + 1) if idx._payload else 0
        idx._index.load_index(index_path)
        return idx, embed_model


class BruteForceIndex:
    """
    Dependency-free vector "DB": stores full matrix and does cosine by dot product.
    Files:
      - vectors.npy
      - payload.jsonl
      - meta.json
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None  # shape (n, dim), float32, normalized
        self._payload: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, payloads: List[Dict[str, Any]]) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Bad vectors shape: {vectors.shape}, expected (*, {self.dim})")
        if len(payloads) != vectors.shape[0]:
            raise ValueError("vectors and payloads length mismatch")
        vectors = np.asarray(vectors, dtype=np.float32)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self._payload.extend(payloads)

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[float, Dict[str, Any]]]:
        if self._vectors is None or not self._payload:
            return []
        q = query_vec.astype(np.float32)
        if q.ndim == 2:
            q = q[0]
        scores = self._vectors @ q  # cosine if vectors are normalized
        k = min(int(k), int(scores.shape[0]))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        out: List[Tuple[float, Dict[str, Any]]] = []
        for i in idx.tolist():
            out.append((float(scores[i]), self._payload[int(i)]))
        return out

    def save(self, out_dir: str, embed_model: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        vectors_path = os.path.join(out_dir, "vectors.npy")
        payload_path = os.path.join(out_dir, "payload.jsonl")
        meta_path = os.path.join(out_dir, "meta.json")

        if self._vectors is None:
            raise ValueError("No vectors to save")
        np.save(vectors_path, self._vectors)
        with open(payload_path, "w", encoding="utf-8") as f:
            for p in self._payload:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        meta = {"kind": "bruteforce", "dim": self.dim, "embed_model": embed_model}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, out_dir: str) -> Tuple["BruteForceIndex", str]:
        vectors_path = os.path.join(out_dir, "vectors.npy")
        payload_path = os.path.join(out_dir, "payload.jsonl")
        meta_path = os.path.join(out_dir, "meta.json")
        if not os.path.isfile(vectors_path) or not os.path.isfile(payload_path) or not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Index not found in {out_dir}. Expected files: vectors.npy, payload.jsonl, meta.json"
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        dim = int(meta["dim"])
        embed_model = str(meta.get("embed_model", ""))
        idx = cls(dim=dim)
        idx._vectors = np.load(vectors_path).astype(np.float32)
        payload: List[Dict[str, Any]] = []
        with open(payload_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload.append(json.loads(line))
        idx._payload = payload
        return idx, embed_model


AnyIndex = Union[HnswIndex, BruteForceIndex]


def create_best_index(dim: int, prefer_hnsw: bool = True) -> AnyIndex:
    if prefer_hnsw:
        try:
            return HnswIndex(dim=dim)
        except Exception:
            return BruteForceIndex(dim=dim)
    return BruteForceIndex(dim=dim)


def load_best_index(out_dir: str) -> Tuple[AnyIndex, str]:
    # Prefer HNSW if present
    if os.path.isfile(os.path.join(out_dir, "hnsw.index")) and os.path.isfile(os.path.join(out_dir, "meta.json")):
        try:
            return HnswIndex.load(out_dir)
        except Exception:
            pass
    return BruteForceIndex.load(out_dir)


