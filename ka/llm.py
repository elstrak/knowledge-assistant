from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_key: str
    timeout_s: float = 60.0


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def chat(self, system: str, user: str) -> str:
        base = self.cfg.base_url.strip().rstrip("/")

        # Если пользователь случайно указал полный endpoint:
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]

        # Если забыли /v1 
        if not base.endswith("/v1") and "/v1/" not in base:
            base = base + "/v1"

        url = f"{base}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(os.getenv("KA_LLM_TEMPERATURE", "0.2")),
            "max_tokens": int(os.getenv("KA_LLM_MAX_TOKENS", "1200"))
        }

        r = requests.post(url, json=payload, headers=headers, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Unexpected response: {data}")

        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected response: {data}")
        return content


_DEFAULT_LLM: Optional[LLMClient] = None


def get_default_llm() -> Optional[LLMClient]:
    global _DEFAULT_LLM
    if _DEFAULT_LLM is not None:
        return _DEFAULT_LLM

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=os.getenv("KA_DOTENV_PATH", ".env"))
    except Exception:
        pass

    base = os.getenv("KA_LLM_BASE_URL", "https://api.mistral.ai").strip()
    model = os.getenv("KA_LLM_MODEL", "mistral-large-latest").strip()
    api_key = os.getenv("KA_LLM_API_KEY", os.getenv("KA_LLM_API_KEY", "")).strip()

    if not api_key:
        raise RuntimeError(
            "LLM API key не задан. Укажи KA_LLM_API_KEY в .env."
            "Положи ключ в .env и загрузай через python-dotenv или export переменную окружения."
        )

    timeout_s = float(os.getenv("KA_LLM_TIMEOUT_S", "60"))
    _DEFAULT_LLM = LLMClient(LLMConfig(model=model, base_url=base, api_key=api_key, timeout_s=timeout_s))
    return _DEFAULT_LLM
