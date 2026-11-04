import argparse
from contextlib import ExitStack
from pathlib import Path

from rag import Client, Index, MODES, load_config, load_runtime, predict_one, read_jsonl, validate_questions, write_line


def main():
    parser = argparse.ArgumentParser(description="Ответить на вопросы через Chat Completions")
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=MODES, help="По умолчанию mode из config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    runtime = load_runtime(config["runtime_path"])
    mode = args.mode or config["mode"]
    questions = list(read_jsonl(args.questions))
    validate_questions(questions)
    inputs = {Path(path).resolve() for path in (args.questions, args.config, config["runtime_path"])}
    outputs = [Path(args.output).resolve(), Path(args.trace).resolve()]
    if len(set(outputs)) != 2 or any(path in inputs or path.exists() for path in outputs):
        parser.error("Выходные файлы должны различаться и ещё не существовать")
    index = Index(args.index)
    client = None
    try:
        for key in ("chunk_chars", "chunk_overlap"):
            if index.metadata[key] != str(config[key]):
                parser.error(f"Индекс построен с другим {key}")
        client = Client(runtime)
        with ExitStack() as stack:
            streams = []
            for path in outputs:
                path.parent.mkdir(parents=True, exist_ok=True)
                streams.append(stack.enter_context(path.open("x", encoding="utf-8")))
            answers, traces = streams
            for question in questions:
                answer, trace = predict_one(index, client, question, config, mode)
                trace["model"] = client.model
                trace["model_revision"] = runtime["revision"]
                trace["generation"] = runtime["generation"]
                trace["orchestrator"] = "langchain"
                trace["corpus_sha256"] = index.metadata["corpus_sha256"]
                trace["config"] = {key: value for key, value in config.items() if key != "runtime_path"}
                write_line(answers, answer)
                write_line(traces, trace)
    finally:
        index.close()
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
