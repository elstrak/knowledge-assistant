"""Formatting utilities for Telegram messages."""
from typing import List
from ka.retriever import RetrievalHit


def escape_markdown(text: str) -> str:
    """Escape markdown special characters for Telegram MarkdownV2."""
    # Для простоты используем HTML режим, но оставляем функцию на будущее
    return text


def format_answer_for_telegram(answer: str) -> str:
    """
    Format RAG answer for Telegram.
    
    - Clean up excessive newlines
    - Convert markdown code blocks to HTML
    - Escape HTML special characters
    """
    import re
    import logging
    logger = logging.getLogger(__name__)
    
    # Логируем исходный ответ для отладки
    logger.debug(f"Исходный ответ (первые 500 символов): {answer[:500]}")
    
    # Конвертируем markdown код блоки (```language ... ```) в HTML <pre><code>
    # Паттерн: ```language\ncode\n```
    code_block_pattern = re.compile(
        r'```(\w+)?\n(.*?)```',
        re.DOTALL
    )
    
    def replace_code_block(match):
        lang = match.group(1) or ""
        code = match.group(2).strip()
        # Экранируем HTML в коде
        code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Telegram не поддерживает атрибут class в тегах code/pre
        return f'<pre><code>{code_escaped}</code></pre>'
    
    result = code_block_pattern.sub(replace_code_block, answer)
    
    # Конвертируем inline код (`code`) в HTML <code>
    inline_code_pattern = re.compile(r'`([^`]+)`')
    def replace_inline_code(match):
        code = match.group(1)
        # Экранируем HTML
        code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<code>{code_escaped}</code>'
    
    result = inline_code_pattern.sub(replace_inline_code, result)
    
    # Удаляем все атрибуты из ВСЕХ HTML тегов (не только code/pre)
    # Telegram HTML поддерживает только определённый набор тегов БЕЗ атрибутов
    result = re.sub(r'<(\w+)\s+[^>]*>', r'<\1>', result, flags=re.IGNORECASE)
    
    # Логируем после очистки
    logger.debug(f"После очистки (первые 500 символов): {result[:500]}")
    
    # Конвертируем markdown заголовки в HTML (опционально, но лучше оставить как есть)
    # result = re.sub(r'^### (.*)$', r'<b>\1</b>', result, flags=re.MULTILINE)
    # result = re.sub(r'^## (.*)$', r'<b>\1</b>', result, flags=re.MULTILINE)
    # result = re.sub(r'^# (.*)$', r'<b>\1</b>', result, flags=re.MULTILINE)
    
    # Убираем лишние переносы
    lines = result.split("\n")
    cleaned = []
    prev_empty = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not prev_empty:
                cleaned.append("")
                prev_empty = True
        else:
            cleaned.append(line)
            prev_empty = False
    
    result = "\n".join(cleaned).strip()
    return result


def split_long_message(text: str, max_length: int = 4000) -> List[str]:
    """
    Split long message into chunks for Telegram (max 4096 chars).
    
    Args:
        text: Text to split
        max_length: Maximum length per chunk (leave some margin)
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks: List[str] = []
    current = ""
    
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_length:
            if current:
                chunks.append(current.strip())
            current = line
        else:
            current += "\n" + line if current else line
    
    if current:
        chunks.append(current.strip())
    
    return chunks


def format_sources_compact(hits: List[RetrievalHit], max_sources: int = 5) -> str:
    """Format sources in a compact way for Telegram."""
    if not hits:
        return ""
    
    lines = ["", "<b>📚 Источники:</b>"]
    for i, h in enumerate(hits[:max_sources], start=1):
        # Короткий формат: номер) заметка → секция
        title = h.title[:50] + "..." if len(h.title) > 50 else h.title
        section = h.section[:30] + "..." if len(h.section) > 30 else h.section
        lines.append(f"{i}) {title} → {section}")
    
    return "\n".join(lines)

