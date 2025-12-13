import os
import sys
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ka.jsonl import read_jsonl
from ka.rag import Retriever


@dataclass
class Metrics:
    n: int
    recall_at_k: float
    mrr: float


def _rank_of_expected(hits: List[Tuple[float, Dict[str, object]]], expected_note_id: str, expected_chunk_id: str) -> Optional[int]:
    """
    Return 1-based rank if found either expected_chunk_id OR expected_note_id, else None.
    We count chunk match as stronger, but for baseline we accept note match too.
    """
    for i, (_score, p) in enumerate(hits, start=1):
        cid = str(p.get("chunk_id", ""))
        nid = str(p.get("note_id", ""))
        if expected_chunk_id and cid == expected_chunk_id:
            return i
        if expected_note_id and nid == expected_note_id:
            return i
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics on validation set")
    parser.add_argument("--index", default="dataset/index", help="Index dir (default: dataset/index)")
    parser.add_argument(
        "--val",
        default="dataset/validation/validation.jsonl",
        help="Validation jsonl (default: dataset/validation/validation.jsonl)",
    )
    parser.add_argument("--k", type=int, default=5, help="k for Recall@k (default: 5)")
    args = parser.parse_args()

    index_dir = os.path.abspath(os.path.expanduser(args.index))
    val_path = os.path.abspath(os.path.expanduser(args.val))

    retriever = Retriever(index_dir=index_dir)
    val_rows = list(read_jsonl(val_path))
    if not val_rows:
        raise SystemExit(f"Empty validation set: {val_path}")

    found = 0
    rr_sum = 0.0

    # We need raw payloads with ids for metrics; easiest: use internal index.search via private access.
    # So we reimplement retrieval using Retriever but also keep note/chunk ids already.
    # We'll just call Retriever.retrieve and compute ranks from those hits.
    for row in val_rows:
        q = str(row.get("query", ""))
        exp_note = str(row.get("expected_note_id", ""))
        exp_chunk = str(row.get("expected_chunk_id", ""))
        hits = retriever.retrieve(q, k=args.k)

        rank = None
        for i, h in enumerate(hits, start=1):
            if exp_chunk and h.chunk_id == exp_chunk:
                rank = i
                break
            if exp_note and h.note_id == exp_note:
                rank = i
                break

        if rank is not None:
            found += 1
            rr_sum += 1.0 / float(rank)

    n = len(val_rows)
    metrics = Metrics(
        n=n,
        recall_at_k=float(found) / float(n),
        mrr=float(rr_sum) / float(n),
    )

    print("=== Retrieval metrics (baseline) ===")
    print(f"N: {metrics.n}")
    print(f"Recall@{args.k}: {metrics.recall_at_k:.4f}")
    print(f"MRR@{args.k}: {metrics.mrr:.4f}")


if __name__ == "__main__":
    main()


