import os
import sys
import argparse
import json
import re
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ka.jsonl import read_jsonl, write_jsonl
from ka.retriever import Retriever, _tokens
from ka.generator import answer_with_llm
from ka.llm import get_default_llm


BRACKET_CIT_RE = re.compile(r"\[([^\]]+)\]")
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i)]
    return [str(x)] if str(x) else []


def get_relevants(ex: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    rel_chunks = as_list(ex.get("relevant_chunk_ids")) or as_list(ex.get("expected_chunk_id"))
    rel_notes = as_list(ex.get("relevant_note_ids")) or as_list(ex.get("expected_note_id"))
    if not rel_notes and rel_chunks:
        rel_notes = list({c.split("#", 1)[0] for c in rel_chunks if c})
    return rel_chunks, rel_notes


def first_rank_match(items: List[str], ranked: List[str], k: int) -> Optional[int]:
    s = set(items)
    for i, x in enumerate(ranked[:k], start=1):
        if x in s:
            return i
    return None


def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def context_coverage(answer: str, context: str) -> float:
    at = [t for t in _tokens(answer) if t]
    ct = set(_tokens(context))
    if not at:
        return 0.0
    in_ctx = sum(1 for t in at if t in ct)
    return in_ctx / len(at)


def extract_bracket_citations(answer: str) -> List[str]:
    return [m.group(1).strip() for m in BRACKET_CIT_RE.finditer(answer or "") if m.group(1).strip()]


def strip_sources_block(answer: str) -> str:
    return (answer or "").split("Источники:", 1)[0].strip()

def parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def judge_prompt(question: str, context: str, answer: str) -> Tuple[str, str]:
    system = (
        "Ты строгий ассистент-оценщик качества ответа в RAG-системе. "
        "Оцени только по предоставленному контексту. "
        "Если ответа нет в контексте, это должно снижать оценку groundedness и correctness."
    )
    user = f"""
Вопрос:
{question}

Контекст (фрагменты заметок):
{context}

Ответ модели:
{answer}

Верни ТОЛЬКО JSON без пояснений, вида:
{{
  "correctness": 1-5,
  "groundedness": 1-5,
  "uses_context": true/false,
  "hallucination": true/false,
  "short_reason": "очень коротко (до 20 слов)"
}}

Правила:
- correctness: насколько ответ отвечает на вопрос (даже если контекст слабый).
- groundedness: насколько ответ опирается на контекст и не додумывает.
- hallucination=true, если есть факты, которых нет в контексте.
- uses_context=true, если в ответе явно используются сведения из контекста.
""".strip()
    return system, user


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate RAG end-to-end on manual validation set.")
    p.add_argument("--index", default="dataset/index")
    p.add_argument("--validation", default="dataset/validation/rag_validation_gc.jsonl")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max_n", type=int, default=0)

    p.add_argument("--judge", action="store_true", help="Enable LLM-as-a-judge scoring")
    p.add_argument("--judge_n", type=int, default=0, help="Judge only first N examples (0 = all)")

    p.add_argument("--report", default="dataset/validation/rag_eval_report.jsonl", help="Write per-example report JSONL")
    args = p.parse_args()

    retriever = Retriever(index_dir=os.path.abspath(os.path.expanduser(args.index)))
    rows = list(read_jsonl(os.path.abspath(os.path.expanduser(args.validation))))
    if args.max_n and args.max_n > 0:
        rows = rows[: args.max_n]
    if not rows:
        raise SystemExit("Empty validation set")

    llm_judge = get_default_llm() if args.judge else None

    ks = [1, 3, 5, 10]
    ks = [k for k in ks if k <= max(1, args.k)]
    max_k = max(ks) if ks else max(1, args.k)

    # Retrieval metrics (chunk + note)
    recall_chunk = {k: 0 for k in ks}
    recall_note = {k: 0 for k in ks}
    mrr_chunk = {k: [] for k in ks}
    mrr_note = {k: [] for k in ks}

    # Generation metrics (no reference)
    n = len(rows)
    got_any_hits = 0
    cited_any = 0
    cited_valid = 0
    avg_cov = 0.0

    # Judge metrics
    judge_count = 0
    judge_correctness: List[float] = []
    judge_groundedness: List[float] = []
    judge_hallucination_rate = 0

    report_rows: List[Dict[str, Any]] = []

    for idx, ex in enumerate(rows):
        q = str(ex.get("query", "") or "")
        rel_chunks, rel_notes = get_relevants(ex)

        hits = retriever.retrieve(q, k=max_k)
        got_chunks = [h.chunk_id for h in hits]
        got_notes = [h.note_id for h in hits]

        if hits:
            got_any_hits += 1

        # retrieval metrics
        for k in ks:
            r_chunk = first_rank_match(rel_chunks, got_chunks, k) if rel_chunks else None
            if r_chunk is not None:
                recall_chunk[k] += 1
                mrr_chunk[k].append(1.0 / r_chunk)
            else:
                mrr_chunk[k].append(0.0)

            r_note = first_rank_match(rel_notes, got_notes, k) if rel_notes else None
            if r_note is not None:
                recall_note[k] += 1
                mrr_note[k].append(1.0 / r_note)
            else:
                mrr_note[k].append(0.0)

        # generation
        context = "\n".join([h.text for h in hits])
        ans = answer_with_llm(q, hits)
        ans_body = strip_sources_block(ans)

        # citations (в теле)
        cits = extract_bracket_citations(ans_body)
        if cits:
            cited_any += 1

        # valid citation: хотя бы одна ссылочная строка содержит chunk_id/note_id из retrieved
        hit_ids = set([h.chunk_id for h in hits] + [h.note_id for h in hits])
        is_valid_cit = any(any(hid in cit for hid in hit_ids) for cit in cits)
        if is_valid_cit:
            cited_valid += 1

        # groundedness proxy: coverage
        cov = context_coverage(ans_body, context)
        avg_cov += cov

        # judge
        judge_out = None
        if llm_judge and (args.judge_n == 0 or judge_count < args.judge_n):
            sys_prompt, user_prompt = judge_prompt(q, context, ans_body)
            raw = llm_judge.chat(system=sys_prompt, user=user_prompt)
            judge_out = parse_json_loose(raw)
            if judge_out:
                judge_count += 1
                c = float(judge_out.get("correctness", 0) or 0)
                g = float(judge_out.get("groundedness", 0) or 0)
                judge_correctness.append(c)
                judge_groundedness.append(g)
                if bool(judge_out.get("hallucination", False)):
                    judge_hallucination_rate += 1

        report_rows.append(
            {
                "query": q,
                "relevant_chunk_ids": rel_chunks,
                "relevant_note_ids": rel_notes,
                "retrieved_chunk_ids": got_chunks[: max_k],
                "retrieved_note_ids": got_notes[: max_k],
                "answer": ans,
                "answer_body": ans_body,
                "coverage": cov,
                "has_bracket_citations": bool(cits),
                "has_valid_bracket_citation": bool(is_valid_cit),
                "judge": judge_out,
            }
        )

    # write report
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    write_jsonl(os.path.abspath(args.report), report_rows)

    print(f"[OK] Evaluated {n} examples")
    print(f"Report saved to: {args.report}\n")

    # retrieval summary
    print("Retriever metrics:")
    for k in ks:
        print(
            f"  k={k:>2} | "
            f"Recall@k(chunks)={(recall_chunk[k]/n):.3f}  MRR@k(chunks)={mean(mrr_chunk[k]):.3f} | "
            f"Recall@k(notes)={(recall_note[k]/n):.3f}   MRR@k(notes)={mean(mrr_note[k]):.3f}"
        )

    # generation summary
    print("\nGenerator (no-reference) metrics:")
    print(f"  retrieval hit rate (>=1 chunk): {got_any_hits/n:.3f}")
    print(f"  bracket citations present:      {cited_any/n:.3f}")
    print(f"  valid bracket citation present: {cited_valid/n:.3f}")
    print(f"  avg token coverage in context:  {avg_cov/n:.3f}")

    # judge summary
    if llm_judge:
        if judge_count == 0:
            print("\nJudge: no parsed results.")
        else:
            print("\nLLM-as-a-judge metrics:")
            print(f"  judged: {judge_count}")
            print(f"  avg correctness (1-5):  {mean(judge_correctness):.3f}")
            print(f"  avg groundedness (1-5): {mean(judge_groundedness):.3f}")
            print(f"  hallucination rate:     {(judge_hallucination_rate/judge_count):.3f}")


if __name__ == "__main__":
    main()
