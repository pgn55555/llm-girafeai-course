import argparse
import json
import statistics
from pathlib import Path

from rag import Index, read_jsonl, validate_answer, validate_questions


CATEGORIES = ("conditions", "comparison", "computation", "insufficient")


def by_id(rows, label):
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError(f"{label}: повтор id={row['id']}")
        result[row["id"]] = row
    return result


def validate_gold(gold, question, corpus_ids):
    if gold.get("category") not in CATEGORIES:
        raise ValueError(f"{question['id']}: неизвестная категория gold")
    if type(gold.get("abstain")) is not bool or not isinstance(gold.get("answer_ids"), list):
        raise ValueError(f"{question['id']}: неверный gold")
    ids = gold["answer_ids"]
    if any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("gold.answer_ids должен содержать уникальные строки")
    if not set(ids) <= {option["id"] for option in question["options"]}:
        raise ValueError("gold содержит неизвестный вариант")
    if gold["abstain"]:
        if ids or gold["category"] != "insufficient":
            raise ValueError("Неверная категория или варианты отказа в gold")
        return
    if not ids or gold["category"] == "insufficient":
        raise ValueError("Неверные варианты разрешимого вопроса в gold")
    sets = gold.get("citation_sets")
    if not isinstance(sets, list) or not sets:
        raise ValueError(f"{question['id']}: заполните citation_sets реальными block_id корпуса")
    for group in sets:
        if not isinstance(group, list) or not 1 <= len(group) <= 4:
            raise ValueError("Набор эталонных ссылок должен содержать 1–4 block_id")
        if any(not isinstance(value, str) for value in group) or len(set(group)) != len(group):
            raise ValueError("Эталонные ссылки должны быть уникальными строками")
        if not set(group) <= corpus_ids:
            raise ValueError("В gold есть block_id вне корпуса")


def citation_f1(predicted, gold_sets):
    predicted = set(predicted)
    return max(2 * len(predicted & set(group)) / (len(predicted) + len(set(group)))
               for group in gold_sets)


def evaluate(questions, gold_rows, answers, traces, corpus_ids, no_context=False):
    validate_questions(questions)
    gold = by_id(gold_rows, "gold")
    predictions = by_id(answers, "answers")
    trace_map = by_id(traces, "trace")
    ids = {question["id"] for question in questions}
    if not ids or set(gold) != ids or set(predictions) != ids or set(trace_map) != ids:
        raise ValueError("Наборы ID questions, gold, answers, trace должны совпадать и быть непустыми")
    scores = {category: [] for category in CATEGORIES}
    accuracy = []
    errors = []
    efficiency_valid = True
    for question in questions:
        qid = question["id"]
        reference, answer, trace = gold[qid], predictions[qid], trace_map[qid]
        validate_gold(reference, question, corpus_ids)
        invalid = None
        try:
            allowed = {item["block_id"] for item in trace.get("context", [])} & corpus_ids
            validate_answer(answer, question, allowed, no_context)
        except (ValueError, KeyError, TypeError) as error:
            invalid = str(error)
        for field in ("llm_calls", "input_tokens", "output_tokens"):
            if type(trace.get(field)) is not int or trace[field] < 0:
                raise ValueError(f"{qid}: неверный счётчик trace.{field}")
        if not isinstance(trace.get("elapsed_ms"), (int, float)) or trace["elapsed_ms"] < 0:
            raise ValueError(f"{qid}: неверное время в trace")
        limit_bad = (trace.get("limit_exceeded", False) or trace["llm_calls"] > 4 or
                     trace["input_tokens"] > 16384 or trace["output_tokens"] > 4096)
        if limit_bad:
            invalid = "Превышение бюджета"
        if not trace.get("usage_verified", False) or limit_bad:
            efficiency_valid = False
        
        if trace.get("error"):
            invalid = trace["error"]
        correct = (not invalid and answer["abstain"] == reference["abstain"] and
                   set(answer["answer_ids"]) == set(reference["answer_ids"]))
        accuracy.append(int(correct))
        if reference["abstain"]:
            score = float(correct)
        elif correct and not no_context:
            score = 0.8 + 0.2 * citation_f1(answer["citations"], reference["citation_sets"])
        else:
            score = 0.0
        scores[reference["category"]].append(score)
        if not correct or (not no_context and score < 1):
            errors.append({"id": qid, "question": question, "gold": reference,
                           "answer": answer, "score": score, "technical_error": invalid,
                           "retrieved_block_ids": trace["retrieved_block_ids"],
                           "context": trace.get("context", [])})
    if any(not values for values in scores.values()):
        raise ValueError("Для Q нужны вопросы всех четырёх категорий")
    rows = list(trace_map.values())
    mean = statistics.mean
    metrics = {"answer_accuracy": mean(accuracy),
               "Q": None if no_context else mean(mean(values) for values in scores.values()),
               "mean_input_tokens": mean(row["input_tokens"] for row in rows),
               "mean_output_tokens": mean(row["output_tokens"] for row in rows),
               "mean_llm_calls": mean(row["llm_calls"] for row in rows),
               "mean_cost": mean(row["input_tokens"] + 4 * row["output_tokens"] +
                                 256 * row["llm_calls"] for row in rows),
               "median_latency_ms": statistics.median(row["elapsed_ms"] for row in rows),
               "efficiency_valid": efficiency_valid,
               "category_scores": None if no_context else {key: mean(value) for key, value in scores.items()}}
    return metrics, errors


def main():
    parser = argparse.ArgumentParser(description="Локальная оценка формата, вариантов, ссылок и расхода")
    for name in ("questions", "gold", "answers", "trace", "index", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()
    index = Index(args.index)
    try:
        metrics, errors = evaluate(
            list(read_jsonl(args.questions)), list(read_jsonl(args.gold)),
            list(read_jsonl(args.answers)), list(read_jsonl(args.trace)),
            index.block_ids(), args.no_context
        )
    finally:
        index.close()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"metrics": metrics, "errors": errors}, stream, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
