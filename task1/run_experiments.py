import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from evaluate import CATEGORIES, by_id, evaluate, validate_gold
from rag import Index, MODES, load_config, load_runtime, read_jsonl, validate_questions


COLUMNS = ["name", "answer_accuracy", "Q", "mean_input_tokens", "mean_output_tokens",
           "mean_llm_calls", "mean_cost", "median_latency_ms"]


def main():
    parser = argparse.ArgumentParser(description="Три системы и две независимые абляции")
    for name in ("index", "questions", "gold", "config", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    load_runtime(config["runtime_path"])
    questions = list(read_jsonl(args.questions))
    gold = list(read_jsonl(args.gold))
    validate_questions(questions)
    references = by_id(gold, "gold")

    index = Index(args.index)
    try:
        corpus_ids = index.block_ids()
    finally:
        index.close()
    for question in questions:
        validate_gold(references[question["id"]], question, corpus_ids)
    
    output = Path(args.output).resolve()

    output.mkdir(parents=True)
    shutil.copyfile(args.config, output / "config.snapshot.json")
    
    runtime = load_runtime(config["runtime_path"])
    with (output / "generation.snapshot.json").open("w", encoding="utf-8") as stream:
        json.dump({key: runtime[key] for key in ("model", "revision", "generation", "timeout_seconds")},
                  stream, ensure_ascii=False, indent=2)
    rows = []
    all_errors = {}
    all_metrics = {}
    for mode in MODES:
        answers = output / f"{mode}_answers.jsonl"
        trace = output / f"{mode}_trace.jsonl"
        command = [sys.executable, str(Path(__file__).with_name("predict.py")),
                   "--index", str(Path(args.index).resolve()),
                   "--questions", str(Path(args.questions).resolve()),
                   "--config", str(Path(args.config).resolve()),
                   "--output", str(answers), "--trace", str(trace), "--mode", mode]
        subprocess.run(command, check=True)
        metrics, errors = evaluate(questions, gold, list(read_jsonl(answers)),
                                   list(read_jsonl(trace)), corpus_ids, mode == "no_context")
        rows.append({"name": mode, **{key: metrics[key] for key in COLUMNS[1:]}})
        all_errors[mode] = errors
        all_metrics[mode] = metrics
    baseline = all_metrics["baseline"]["mean_cost"]
    for mode, metrics in all_metrics.items():
        if mode == "no_context":
            continue
        verified = metrics["efficiency_valid"] and all_metrics["baseline"]["efficiency_valid"]
        efficiency = min(1, baseline / max(metrics["mean_cost"], 1)) if verified else 0
        metrics["E"] = efficiency
        metrics["total"] = 75 * metrics["Q"] + 25 * metrics["Q"] * efficiency
    with (output / "experiments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(all_metrics, stream, ensure_ascii=False, indent=2)
    with (output / "errors.json").open("w", encoding="utf-8") as stream:
        json.dump(all_errors, stream, ensure_ascii=False, indent=2)
    shutil.copyfile(output / "final_answers.jsonl", output / "dev_answers.jsonl")
    shutil.copyfile(output / "final_trace.jsonl", output / "dev_trace.jsonl")


if __name__ == "__main__":
    main()
