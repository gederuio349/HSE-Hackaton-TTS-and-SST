# Используем Python 3.10 (требуется для numpy==1.22.0)
FROM python:3.10-slim

# Устанавливаем системные зависимости для обработки аудио и других библиотек
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    sox \
    git \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта
COPY . .

# Создаем необходимые директории
RUN mkdir -p uploads output

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu

# Открываем порт для FastAPI
EXPOSE 4010

# Команда запуска (можно изменить на run_https.py для HTTPS)
CMD ["python", "api.py"]

