import argparse
import hashlib
import json
import os
import platform
import resource
import sqlite3
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import load_config, read_jsonl, terms


def main():
    parser = argparse.ArgumentParser(description="Построить локальный BM25-индекс ТФКП")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_chars"], chunk_overlap=config["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""], add_start_index=True,
        keep_separator=True, strip_whitespace=False
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "index.sqlite"
    if target.exists():
        parser.error("Индекс уже существует. Используйте новую выходную папку.")
    temporary = output / "index.sqlite.building"
    if temporary.exists():
        parser.error("Найден незавершённый индекс. Используйте новую выходную папку.")
    started = time.perf_counter()
    db = sqlite3.connect(temporary)
    try:
        db.executescript("""
            CREATE TABLE blocks (
                block_id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE,
                page INTEGER, section TEXT, text TEXT
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, block_id TEXT, page INTEGER,
                section TEXT, start INTEGER, text TEXT
            );
            CREATE VIRTUAL TABLE search USING fts5(terms, tokenize='ascii');
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        """)
        count = 0
        chunk_count = 0
        for ordinal, block in enumerate(read_jsonl(args.corpus)):
            bid, page, section, text = (block[key] for key in ("block_id", "page", "section", "text"))
            if not isinstance(bid, str) or not bid or type(page) is not int or page < 1:
                raise ValueError(f"Строка {ordinal + 1}: неверные block_id/page")
            if not isinstance(text, str) or not text.strip() or not isinstance(section, str):
                raise ValueError(f"{bid}: нужны непустой text и строка section")
            db.execute("INSERT INTO blocks VALUES (?, ?, ?, ?, ?)", (bid, ordinal, page, section, text))
            for document in splitter.create_documents([text]):
                start = document.metadata["start_index"]
                part = document.page_content
                if start < 0 or text[start:start + len(part)] != part:
                    raise ValueError(f"{bid}: не удалось сохранить смещение чанка")
                cursor = db.execute(
                    "INSERT INTO chunks(block_id,page,section,start,text) VALUES (?,?,?,?,?)",
                    (bid, page, section, start, part)
                )
                db.execute("INSERT INTO search(rowid,terms) VALUES (?,?)", (cursor.lastrowid, " ".join(terms(part))))
                chunk_count += 1
            count += 1
        if not count:
            raise ValueError("Корпус пуст")
        with open(args.corpus, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = {"version": "2", "splitter": "RecursiveCharacterTextSplitter",
                    "corpus_sha256": digest,
                    "chunk_chars": str(config["chunk_chars"]),
                    "chunk_overlap": str(config["chunk_overlap"])}
        db.executemany("INSERT INTO metadata VALUES (?,?)", metadata.items())
        db.execute("INSERT INTO search(search) VALUES ('optimize')")
        db.commit()
    finally:
        db.close()
    os.replace(temporary, target)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats = {"blocks": count, "chunks": chunk_count, "corpus_sha256": digest,
             "build_seconds": round(time.perf_counter() - started, 3),
             "peak_ram_bytes": peak if platform.system() == "Darwin" else peak * 1024,
             "index_bytes": target.stat().st_size, "python": platform.python_version(),
             "platform": platform.platform(), "cpu": platform.processor()}
    with open(output / "build_stats.json", "w", encoding="utf-8") as stream:
        json.dump(stats, stream, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
