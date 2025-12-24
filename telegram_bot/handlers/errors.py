"""Error handlers."""
import logging
from aiogram import Router
from aiogram.types import ErrorEvent

router = Router()
logger = logging.getLogger(__name__)


@router.error()
async def error_handler(event: ErrorEvent) -> None:
    """Handle all unhandled errors."""
    logger.error(f"Unhandled error: {event.exception}", exc_info=True)
    
    if event.update.message:
        try:
            await event.update.message.answer(
                "❌ <b>Произошла непредвиденная ошибка.</b>\n\n"
                "Попробуйте позже или обратитесь к администратору.",
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")

