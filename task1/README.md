# RAG по учебнику ТФКП

**Автор:** Иванов Максим Олегович. **Группа:** М05-602а.

## Подготовка

Команды выполняются из корня:

```bash
cd task1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Настройка рантайма

В `config.json` путь `runtime_path` разрешается относительно файла конфигурации. Ожидается локальный `runtime.json` с полями адаптера:

- `model`: `Qwen/Qwen3.5-4B`
- `revision`: точный 40-символьный commit хэш из выданного runtime
- `timeout_seconds`: сетевой тайм-аут;
- `generation`: параметры тела Chat Completions из выданного runtime, включая семплирование, лимит выхода и рассуждения;
- `local_server`: настройки локальных весов — `dtype`, `quantization`, `kv_cache_dtype`. Для проверки на внешнем endpoint это поле не требуется.

Шаблон: [runtime.example.json](runtime.example.json). Скопируйте его в `task1/runtime.json` и перенесите все нужные параметры генерации без изменения значений.

## Локальная Qwen

В `config.json`, раздел `local_llm`, задаются папка весов (относительно config), версия образа, порт, доля GPU-памяти и число GPU.
Для BF16 только 4 млрд параметров занимают примерно 8 GB, дополнительно нужны
память исполнения и кеш.

Сначала заполните runtime, затем отдельно скачайте образ и веса точной ревизии:

```bash
python local_llm.py download --config config.json
```

В первом терминале запустите локальный сервер:

```bash
export LLM_MODEL='Qwen/Qwen3.5-4B'
export LLM_API_KEY='local-tfkp'
python local_llm.py serve --config config.json
```

Скрипт открывает только `127.0.0.1:8000`, читает веса из папки с проверенным манифестом и включает offline-режим Transformers.

Во втором терминале перейдите в `task1`, активируйте `.venv` и задайте:

```bash
export LLM_BASE_URL='http://127.0.0.1:8000/v1'
export LLM_MODEL='Qwen/Qwen3.5-4B'
export LLM_API_KEY='local-tfkp'
export SUBMISSION_ID='YOUR_SUBMISSION_ID'
curl -fsS http://127.0.0.1:8000/health
```

Дождитесь успешного `/health` после загрузки, затем запускайте предсказание.

## Сборка и предсказание

```bash
python build_index.py --corpus /absolute/path/to/data/corpus.jsonl --output ./index --config config.json
python predict.py --index ./index --questions /absolute/path/to/data/dev_questions.jsonl --output ./dev_answers.jsonl --trace ./dev_trace.jsonl --config config.json
```

## Эксперименты

| Режим | Поиск | Соседи |
| --- | --- | --- |
| `no_context` | нет | нет |
| `baseline` | один BM25 по вопросу и вариантам | нет |
| `final` | общий и отдельные запросы, RRF | да |
| `no_option_search` | один общий BM25 | да |
| `no_neighbors` | общий и отдельные запросы, RRF | нет |

Запустить эксперименты:

```bash
python run_experiments.py --index ./index --questions /absolute/path/to/data/dev_questions.jsonl --gold /absolute/path/to/data/dev_gold.local.jsonl --config config.json --output ./artifacts/dev-v1
```

После успешного прогона:

```bash
cp artifacts/dev-v1/experiments.csv experiments.csv
cp artifacts/dev-v1/dev_answers.jsonl dev_answers.jsonl
cp artifacts/dev-v1/dev_trace.jsonl dev_trace.jsonl
python evaluate.py --questions /absolute/path/to/data/dev_questions.jsonl --gold /absolute/path/to/data/dev_gold.local.jsonl --answers dev_answers.jsonl --trace dev_trace.jsonl --index ./index --output ./artifacts/final-evaluation.json
```