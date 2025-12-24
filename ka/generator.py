import os
from typing import List, Optional

from ka.llm import LLMClient, get_default_llm
from ka.retriever import RetrievalHit

def sources_block(hits: List[RetrievalHit], max_sources: int = 10) -> str:
    """Deterministic sources block appended to the answer."""
#def _format_sources(hits: List[RetrievalHit]) -> str:
    if not hits:
        return ""
    lines = ["", "Источники:"]
    for i, h in enumerate(hits, start=1):
        lines.append(f"{i}) {h.note_id} ({h.chunk_id}) — {h.title} → {h.section}")
    return "\n".join(lines)


def build_llm_context(hits: List[RetrievalHit]) -> str:
    """Compact context for LLM. Truncates by KA_CONTEXT_CHARS."""
    max_chars = int(os.getenv("KA_CONTEXT_CHARS", "12000"))
    parts: List[str] = []
    total = 0
    for i, h in enumerate(hits, start=1):
        block = (
            f"[Source {i}]\n"
            f"note_id: {h.note_id}\n"
            f"chunk_id: {h.chunk_id}\n"
            f"title: {h.title}\n"
            f"section: {h.section}\n"
            f"text: {h.text}\n"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()

# format_answer_extractively
def answer_extractively(question: str, hits: List[RetrievalHit]) -> str:
    """Baseline: show retrieved passages (debug-friendly)."""
    if not hits:
        return "Не нашёл релевантных фрагментов в базе."

    lines: List[str] = []
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Answer (baseline, retrieval-only):")
    lines.append("")

    for i, h in enumerate(hits, start=1):
        lines.append(f"{i}) [{h.title} -> {h.section}] (score={h.score:.3f})")
        lines.append(h.text)
        lines.append(f"   source: {h.note_id} ({h.chunk_id})")
        lines.append("")

    return "\n".join(lines).strip() + sources_block(hits)

# format_answer_llm
def answer_with_llm(question: str, hits: List[RetrievalHit], llm: Optional[LLMClient] = None) -> str:
    """LLM answer grounded in retrieved context. Falls back to extractive on any error."""
    if not hits:
        return "Не нашёл релевантных фрагментов в базе."

    llm = llm or get_default_llm()
    if llm is None:
        return answer_extractively(question, hits)

    system = (
        "Ты — ассистент по базе заметок Obsidian. "
        "Отвечай ТОЛЬКО на основе предоставленного контекста. "
        "Если в контексте нет ответа — так и скажи и предложи, что уточнить. "
        "Не выдумывай факты и не добавляй внешние знания."
        "НЕ добавляй список источников — я добавлю источники сам в конце."
    )

    context = build_llm_context(hits)
    user = f"Вопрос: {question}\n\nКонтекст (используй только его):\n{context}"

    try:
        answer = llm.chat(system=system, user=user)
    except Exception:
        return answer_extractively(question, hits)

    return (answer.strip() + sources_block(hits)).strip()


def format_answer(question: str, hits: List[RetrievalHit]) -> str:
    """
    Main entrypoint.

    mode:
      - llm (default)
      - extractive
    """
    mode = os.getenv("KA_ANSWER_MODE", "llm").strip().lower()
    if mode in ("extractive", "baseline"):
        return answer_extractively(question, hits)
    return answer_with_llm(question, hits)