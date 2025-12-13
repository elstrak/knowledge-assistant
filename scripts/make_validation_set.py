import os
import sys
import argparse
import random
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ka.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create weakly-supervised validation set from chunks.jsonl")
    parser.add_argument(
        "--chunks",
        default="dataset/processed/chunks.jsonl",
        help="Path to chunks.jsonl (default: dataset/processed/chunks.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="dataset/validation/validation.jsonl",
        help="Output validation jsonl (default: dataset/validation/validation.jsonl)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of validation examples (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chunks_path = os.path.abspath(os.path.expanduser(args.chunks))
    out_path = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = list(read_jsonl(chunks_path))
    if not rows:
        raise SystemExit(f"Empty chunks.jsonl: {chunks_path}")

    random.seed(args.seed)
    random.shuffle(rows)
    rows = rows[: min(args.n, len(rows))]

    examples: List[Dict[str, object]] = []
    for r in rows:
        title = str(r.get("title", "") or "")
        section = str(r.get("section", "") or "")
        text = str(r.get("text", "") or "")
        note_id = str(r.get("note_id", "") or "")
        chunk_id = str(r.get("chunk_id", "") or "")

        # Weak supervision:
        # - query is derived from metadata; expected is to retrieve the same note/chunk.
        # This is a baseline suitable for checkpoint metrics.
        query = f"{title}. {section}".strip(". ").strip()
        if len(query) < 8:
            query = title or section or (text[:80] if text else "query")

        examples.append(
            {
                "query": query,
                "expected_note_id": note_id,
                "expected_chunk_id": chunk_id,
            }
        )

    write_jsonl(out_path, examples)
    print(f"[INFO] Wrote {len(examples)} validation examples to {out_path}")


if __name__ == "__main__":
    main()


