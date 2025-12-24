"""Bot configuration."""
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.getenv("KA_DOTENV_PATH", ".env"))
except Exception:
    pass


@dataclass
class BotConfig:
    """Configuration for Telegram bot."""
    
    # Telegram
    bot_token: str
    
    # RAG settings
    index_path: str
    notes_path: str
    answer_mode: str
    
    # LLM settings
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    
    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load config from environment variables."""
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not bot_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN не задан. "
                "Создай бота через @BotFather и укажи токен в .env"
            )
        
        return cls(
            bot_token=bot_token,
            index_path=os.getenv("KA_INDEX_PATH", "dataset/index"),
            notes_path=os.getenv("KA_NOTES_PATH", "dataset/processed/notes.jsonl"),
            answer_mode=os.getenv("KA_ANSWER_MODE", "llm").lower(),
            llm_base_url=os.getenv("KA_LLM_BASE_URL", "https://api.mistral.ai"),
            llm_model=os.getenv("KA_LLM_MODEL", "mistral-large-latest"),
            llm_api_key=os.getenv("KA_LLM_API_KEY", ""),
        )

