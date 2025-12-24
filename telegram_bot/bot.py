"""Main bot entry point."""
import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from telegram_bot.config import BotConfig
from telegram_bot.handlers import start, query, errors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Start the bot."""
    try:
        # Load config
        logger.info("Загружаю конфигурацию...")
        config = BotConfig.from_env()
        
        # Initialize bot and dispatcher
        bot = Bot(
            token=config.bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
        dp = Dispatcher()
        
        # Register handlers
        dp.include_router(start.router)
        dp.include_router(query.router)
        dp.include_router(errors.router)
        
        # Initialize retriever
        logger.info("Инициализирую RAG систему...")
        query.init_retriever(
            index_path=config.index_path,
            notes_path=config.notes_path
        )
        
        # Start polling
        logger.info("🚀 Бот запущен и готов к работе!")
        logger.info(f"📊 Режим ответов: {config.answer_mode}")
        logger.info(f"🤖 LLM модель: {config.llm_model}")
        
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
        
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки (Ctrl+C)")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Бот остановлен")


if __name__ == "__main__":
    asyncio.run(main())

