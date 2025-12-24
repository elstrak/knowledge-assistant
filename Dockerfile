# Используем официальный Python образ
FROM python:3.11-slim

# Рабочая директория
WORKDIR /app

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код проекта
COPY ka/ ./ka/
COPY telegram_bot/ ./telegram_bot/
COPY scripts/ ./scripts/

# Создаём директории для данных (будут монтироваться как volumes)
RUN mkdir -p dataset/index dataset/processed

# Запуск бота
CMD ["python", "telegram_bot/bot.py"]

