import os
from typing import List, Optional

from ka.llm import LLMClient, get_default_llm
from ka.retriever import RetrievalHit

def sources_block(hits: List[RetrievalHit], max_sources: int = 10) -> str:
    """Deterministic sources block appended to the answer."""
    if not hits:
        return ""
    lines = ["", "Источники:"]
    for i, h in enumerate(hits, start=1):
        # Экранируем HTML в названиях заметок и разделов
        # Некоторые разделы могут называться <class>, <code>, etc.
        note_id = h.note_id.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        chunk_id = h.chunk_id.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        title = h.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        section = h.section.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        lines.append(f"{i}) {note_id} ({chunk_id}) — {title} → {section}")
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

    import logging
    logger = logging.getLogger(__name__)
    
    llm = llm or get_default_llm()
    if llm is None:
        logger.warning("LLM клиент не инициализирован, используем extractive режим")
        return answer_extractively(question, hits)

    system = (
        "Ты — ассистент по базе заметок Obsidian. "
        "Отвечай на основе предоставленного контекста из заметок. "
        "Используй информацию из контекста для формирования полного и полезного ответа. "
        "Если в контексте есть релевантная информация — используй её. "
        "Если информации недостаточно — скажи что знаешь из контекста и предложи уточнить. "
        "Отвечай на русском языке, структурированно и понятно. "
        "Используй ТОЛЬКО markdown для форматирования (НЕ используй HTML теги). "
        "Для кода используй тройные обратные кавычки ```language ... ```. "
        "НЕ добавляй список источников в конце — источники будут добавлены отдельно."
    )

    context = build_llm_context(hits)
    user = f"Вопрос: {question}\n\nКонтекст из заметок:\n{context}\n\nСформируй полный и полезный ответ на основе этого контекста."

    try:
        logger.info("Отправляю запрос в LLM...")
        logger.debug(f"Контекст для LLM ({len(context)} символов): {context[:200]}...")
        answer = llm.chat(system=system, user=user)
        logger.info(f"Получен ответ от LLM ({len(answer)} символов)")
    except Exception as e:
        logger.error(f"Ошибка при вызове LLM: {e}", exc_info=True)
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