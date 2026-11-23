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

## Проверки и CI

Установите dev-зависимости и запустите локально:

```bash
pip install -e ".[dev]"
pytest -q
ruff check app tests
```

GitHub Actions workflow `.github/workflows/ci.yml` запускается при push в `main`
и `feature/**`, а также при pull request в `main`. Он выполняет полный набор
unit/integration/evaluation tests на Python 3.11 и 3.12, Ruff lint и Docker image
build (без публикации образа).

HTTP integration tests используют подменённый RAG chain: они проверяют маршруты,
валидацию, readiness и сериализацию ответа без скачивания embedding/Cross-Encoder
моделей и без реальных вызовов OpenAI/Mistral. Unit/evaluation tests также должны
оставаться независимыми от секретов провайдера.

## Проверка развернутого экземпляра

После deployment можно проверить доступность API вручную:

```bash
RAG_BASE_URL=https://rag.example.com python scripts/smoke_test.py
```

Или запустить GitHub Actions → **Deployment smoke test** → Run workflow, указав
`base_url`. Проверка ожидает HTTP 200 от `/health/live` и `/health/ready`.
Readiness возвращает 503, пока индекс не восстановлен или не создан — это
ожидаемое поведение пустого экземпляра, а не успешный smoke test.

Workflow smoke test не выполняет загрузку реального документа и генерацию LLM:
для этого нужны отдельный тестовый документ, API key и ключ выбранного LLM provider.
Не храните production-секреты в исходном коде или в workflow inputs.

## Ограничения MVP

Один процесс/worker: параллельные записи и несколько реплик не поддерживаются.
API key общий для сервиса, не заменяет пользовательскую аутентификацию.
Для публичного production добавьте rate limiting, TLS, мониторинг, резервное
копирование volume и ротацию секретов.
