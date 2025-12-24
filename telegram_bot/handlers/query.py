"""Query handling (main RAG logic)."""
import logging
from typing import Optional

from aiogram import Router, F
from aiogram.types import Message
from aiogram.filters import Command

from ka.retriever import Retriever
from ka.generator import format_answer
from ka.jsonl import read_jsonl
from telegram_bot.utils.formatter import (
    format_answer_for_telegram,
    split_long_message,
)

router = Router()
logger = logging.getLogger(__name__)

# Global state (initialized in bot.py)
_retriever: Optional[Retriever] = None
_notes_count: int = 0
_chunks_count: int = 0


def init_retriever(index_path: str, notes_path: str) -> None:
    """Initialize retriever (called once on bot startup)."""
    global _retriever, _notes_count, _chunks_count
    
    logger.info(f"Загружаю индекс из {index_path}...")
    _retriever = Retriever(index_dir=index_path)
    
    # Подсчитываем статистику из индекса (более надёжно)
    try:
        # Получаем payloads из индекса (как в retriever.py)
        from ka.retriever import _iter_index_payloads
        payloads = list(_iter_index_payloads(_retriever._index))
        _chunks_count = len(payloads)
        logger.info(f"Индекс содержит {_chunks_count} чанков")
        
        # Считаем уникальные заметки из чанков
        unique_notes = set()
        for p in payloads:
            note_id = p.get("note_id")
            if note_id:
                unique_notes.add(str(note_id))
        _notes_count = len(unique_notes)
        logger.info(f"Найдено {_notes_count} уникальных заметок в индексе")
    except Exception as e:
        logger.warning(f"Не удалось посчитать статистику из индекса: {e}")
        _chunks_count = 0
        _notes_count = 0
        
        # Fallback: пытаемся загрузить из notes.jsonl
        try:
            notes = list(read_jsonl(notes_path))
            _notes_count = len(notes)
            logger.info(f"Загружено {_notes_count} заметок из {notes_path}")
        except Exception as e2:
            logger.warning(f"Не удалось загрузить заметки из {notes_path}: {e2}")
            _notes_count = 0
    
    logger.info("Retriever готов!")


@router.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    """Show knowledge base statistics."""
    if _retriever is None:
        await message.answer("❌ Индекс не загружен")
        return
    
    await message.answer(
        f"<b>📊 Статистика базы знаний</b>\n\n"
        f"📝 Заметок: {_notes_count}\n"
        f"📦 Чанков: {_chunks_count}\n"
        f"🔍 Режим ответов: LLM-генерация\n\n"
        f"Готов отвечать на вопросы!",
        parse_mode="HTML"
    )


@router.message(F.text)
async def handle_query(message: Message) -> None:
    """Handle text messages as queries."""
    if _retriever is None:
        await message.answer(
            "❌ <b>Ошибка:</b> Индекс не загружен.\n"
            "Обратитесь к администратору бота.",
            parse_mode="HTML"
        )
        return
    
    query = message.text
    if not query or not query.strip():
        return
    
    # Показываем что бот "печатает"
    await message.bot.send_chat_action(message.chat.id, "typing")
    
    try:
        # Retrieval
        logger.info(f"Запрос: {query}")
        hits = _retriever.retrieve(query, k=5)
        
        if not hits:
            await message.answer(
                "🤔 Не нашёл релевантных фрагментов в базе.\n\n"
                "Попробуй:\n"
                "• Переформулировать вопрос\n"
                "• Использовать другие ключевые слова\n"
                "• Уточнить тему",
                parse_mode="HTML"
            )
            return
        
        # Generation
        logger.info(f"Найдено {len(hits)} фрагментов, генерирую ответ...")
        answer = format_answer(query, hits)
        
        # Форматируем и отправляем
        formatted = format_answer_for_telegram(answer)
        chunks = split_long_message(formatted, max_length=4000)
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                await message.answer(chunk, parse_mode="HTML")
            else:
                await message.answer(f"<i>(продолжение)</i>\n\n{chunk}", parse_mode="HTML")
        
        logger.info("Ответ отправлен")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
        await message.answer(
            "❌ <b>Произошла ошибка при обработке запроса.</b>\n\n"
            f"Детали: {str(e)[:200]}\n\n"
            "Попробуй ещё раз или обратись к администратору.",
            parse_mode="HTML"
        )

