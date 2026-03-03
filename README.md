# LinkedIn Icebreaker Bot

Приложение для анализа LinkedIn-профиля и генерации:

- 3 стартовых фактов о человеке
- ответов на вопросы по профилю в формате чата

В проекте используется RAG-пайплайн на базе LlamaIndex:

1. загрузка профиля (ProxyCurl или mock JSON)
2. разбиение данных на ноды
3. построение векторного индекса (OpenAI embeddings)
4. ответы через OpenAI LLM с retrieval по индексу

## Технологии

- Python 3.12
- LlamaIndex (`core`, `llms-openai`, `embeddings-openai`)
- OpenAI API
- Gradio (web UI)
- Requests
- python-dotenv

## Требования

- `OPENAI_API_KEY` обязателен
- `ProxyCurl API Key` нужен только если отключен mock-режим

## Установка

```bash
uv sync
```

## Переменные окружения

Создайте файл `.env` в корне проекта:

```env
OPENAI_API_KEY=your_openai_key
# OPENAI_BASE_URL=
```

Опционально:

- `GRADIO_SERVER_PORT` (например, `7860`)
- `GRADIO_SERVER_NAME` (например, `127.0.0.1`)
- `GRADIO_SHARE=true` для публичной ссылки Gradio

## Запуск

### Web UI

```bash
uv run python app.py
```

Во вкладке `Process LinkedIn Profile`:

- `Use Mock Data = true` -> данные берутся из тестового JSON
- `Use Mock Data = false` -> нужен `ProxyCurl API Key`

### CLI

```bash
uv run python main.py --mock
```

Или с реальным профилем:

```bash
uv run python main.py --url "https://www.linkedin.com/in/username/" --api-key "YOUR_PROXYCURL_KEY"
```

## Конфиг

Основные параметры в `config.py`:

- `LLM_MODEL_ID` (по умолчанию `gpt-4o-mini`)
- `EMBEDDING_MODEL_ID` (по умолчанию `text-embedding-3-small`)
- `SIMILARITY_TOP_K`
- `CHUNK_SIZE`
- шаблоны промптов (`INITIAL_FACTS_TEMPLATE`, `USER_QUESTION_TEMPLATE`)

## Структура проекта

- `app.py` - Gradio интерфейс
- `main.py` - CLI интерфейс
- `modules/data_extraction.py` - загрузка данных профиля (ProxyCurl/mock)
- `modules/data_processing.py` - split в ноды и индексация
- `modules/llm_interface.py` - создание OpenAI LLM/embeddings
- `modules/query_engine.py` - генерация фактов и ответы на вопросы

## Как это работает

1. Профиль загружается как JSON.
2. JSON сериализуется в текст и режется `SentenceSplitter`.
3. Для нод строятся эмбеддинги и `VectorStoreIndex`.
4. На вопрос пользователя берутся релевантные ноды и формируется ответ LLM.

## Troubleshooting

- `Unknown model ...`: выберите модель, поддерживаемую вашей версией `llama-index-llms-openai`.
- `Empty Response`: переключитесь на `gpt-4o-mini` или `gpt-4.1-mini`.
- `Cannot find empty port ...`: задайте другой порт через `GRADIO_SERVER_PORT`.
- Ошибка профиля при `mock=false`: проверьте `ProxyCurl API Key`.
