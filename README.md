# Smart RAG Service

Локальный RAG-сервис: загрузка PDF, гибридный поиск BM25 + FAISS, RRF,
опциональный Cross-Encoder и генерация через OpenAI или Mistral.

## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

В development API key необязателен. Если `API_KEY` задан, защищённые endpoints
требуют заголовок `X-API-Key`.

## Production Docker Compose

1. Создайте `.env` из `.env.example`.
2. Укажите `APP_ENV=production`, `API_KEY` со случайным секретом не короче 32
   символов, ключ LLM и точный список `CORS_ORIGINS`.
3. Запустите `docker compose up --build -d`.

Порт опубликован только на `127.0.0.1:8000`. Для внешнего трафика используйте
TLS reverse proxy. Данные сохраняются в Docker volume `rag_data`.

## Endpoints

- `GET /health/live` — liveness.
- `GET /health/ready` — readiness.
- `GET /status` — состояние (API key).
- `POST /upload` — PDF multipart upload (API key).
- `POST /query` — JSON `{"question":"...","top_k":5}` (API key).
- `DELETE /clear` — очистка in-memory состояния (API key); snapshot на диске остаётся.

Swagger/OpenAPI отключён в production. Загрузка ограничена `MAX_UPLOAD_SIZE_BYTES`,
проверяются расширение и PDF header. Индекс восстанавливается из последнего
атомарного snapshot при старте.

## Ограничения MVP

Один процесс/worker: параллельные записи и несколько реплик не поддерживаются.
API key общий для сервиса, не заменяет пользовательскую аутентификацию.
Для публичного production добавьте rate limiting, TLS, мониторинг, резервное
копирование volume и ротацию секретов.

## Проверки

```bash
pytest
ruff check app tests
```
