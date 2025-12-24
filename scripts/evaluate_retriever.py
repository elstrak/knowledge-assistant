import os
import sys
import argparse
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ka.jsonl import read_jsonl
from ka.retriever import Retriever


def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i)]
    return [str(x)] if str(x) else []


def get_relevants(ex: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    # v2
    rel_chunks = as_list(ex.get("relevant_chunk_ids"))
    rel_notes = as_list(ex.get("relevant_note_ids"))

    # backward-compat v1
    if not rel_chunks:
        rel_chunks = as_list(ex.get("expected_chunk_id"))
    if not rel_notes:
        rel_notes = as_list(ex.get("expected_note_id"))

    # if only chunk_ids provided, derive note_ids
    if not rel_notes and rel_chunks:
        rel_notes = list({c.split("#", 1)[0] for c in rel_chunks if "#" in c or c.endswith(".md")})
    return rel_chunks, rel_notes


def first_rank_match(items: List[str], ranked: List[str], k: int) -> Optional[int]:
    s = set(items)
    for i, x in enumerate(ranked[:k], start=1):
        if x in s:
            return i
    return None


def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate retriever with Recall@k / MRR@k (chunk & note)")
    p.add_argument("--index", default="dataset/index", help="Index directory")
    p.add_argument("--validation", default="dataset/validation/validation.jsonl", help="Validation jsonl")
    p.add_argument("--k", default="1,3,5,10", help="Comma-separated k values")
    p.add_argument("--max_n", type=int, default=0, help="Limit number of examples (0 = all)")
    p.add_argument("--show_errors", type=int, default=5, help="Print N worst/missed examples")
    args = p.parse_args()

    ks = [int(x) for x in args.k.split(",") if x.strip()]
    ks = sorted({k for k in ks if k > 0})
    max_k = max(ks)

    retriever = Retriever(index_dir=os.path.abspath(os.path.expanduser(args.index)))

    rows = list(read_jsonl(os.path.abspath(os.path.expanduser(args.validation))))
    if args.max_n and args.max_n > 0:
        rows = rows[: args.max_n]

    if not rows:
        raise SystemExit("Empty validation set")

    # accumulators
    recall_chunk = {k: 0 for k in ks}
    recall_note = {k: 0 for k in ks}
    mrr_chunk = {k: [] for k in ks}
    mrr_note = {k: [] for k in ks}

    misses: List[Tuple[float, Dict[str, Any], List[str], List[str], List[str], List[str]]] = []
    # (best_score, example, rel_chunks, rel_notes, got_chunks, got_notes)

    for ex in rows:
        q = str(ex.get("query", "") or "")
        rel_chunks, rel_notes = get_relevants(ex)

        hits = retriever.retrieve(q, k=max_k)
        got_chunks = [h.chunk_id for h in hits]
        got_notes = [h.note_id for h in hits]
        best_score = hits[0].score if hits else -999.0

        for k in ks:
            # chunk metrics
            r_chunk = first_rank_match(rel_chunks, got_chunks, k) if rel_chunks else None
            if r_chunk is not None:
                recall_chunk[k] += 1
                mrr_chunk[k].append(1.0 / r_chunk)
            else:
                mrr_chunk[k].append(0.0)

            # note metrics
            r_note = first_rank_match(rel_notes, got_notes, k) if rel_notes else None
            if r_note is not None:
                recall_note[k] += 1
                mrr_note[k].append(1.0 / r_note)
            else:
                mrr_note[k].append(0.0)

        # for diagnostics: “miss at max_k”
        if rel_chunks and first_rank_match(rel_chunks, got_chunks, max_k) is None:
            misses.append((best_score, ex, rel_chunks, rel_notes, got_chunks, got_notes))
        elif (not rel_chunks) and rel_notes and first_rank_match(rel_notes, got_notes, max_k) is None:
            misses.append((best_score, ex, rel_chunks, rel_notes, got_chunks, got_notes))

    n = len(rows)

    print(f"[OK] Evaluated {n} examples on index={args.index}")
    print("")
    for k in ks:
        print(
            f"k={k:>2} | "
            f"Recall@k(chunks)={(recall_chunk[k]/n):.3f}  MRR@k(chunks)={mean(mrr_chunk[k]):.3f} | "
            f"Recall@k(notes)={(recall_note[k]/n):.3f}   MRR@k(notes)={mean(mrr_note[k]):.3f}"
        )

    if args.show_errors and misses:
        print("\n---\nMissed examples (up to max_k):")
        # show the ones where even top score is “confident” but still wrong first
        misses.sort(key=lambda x: x[0], reverse=True)
        for i, (best_score, ex, rel_chunks, rel_notes, got_chunks, got_notes) in enumerate(misses[: args.show_errors], start=1):
            print(f"\n[{i}] score_top1={best_score:.3f}")
            print("query:", ex.get("query"))
            if rel_notes:
                print("expected_notes:", rel_notes)
            if rel_chunks:
                print("expected_chunks:", rel_chunks[:5], ("..." if len(rel_chunks) > 5 else ""))
            print("got_notes:", got_notes[:10])
            print("got_chunks:", got_chunks[:10])


if __name__ == "__main__":
    main()
