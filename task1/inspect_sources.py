import argparse
import json
from pathlib import Path

from rag import Index, read_jsonl


def main():
    parser = argparse.ArgumentParser(description="Показать настоящие блоки для проверки авторских эталонов")
    parser.add_argument("--index", required=True)
    parser.add_argument("--gold", default=str(Path(__file__).with_name("own_gold.jsonl")))
    args = parser.parse_args()
    index = Index(args.index)
    try:
        for reference in read_jsonl(args.gold):
            print(json.dumps({"id": reference["id"], "rationale": reference["rationale"]}, ensure_ascii=False))
            for page in reference["source_pages"]:
                for block in index.db.execute("SELECT block_id,page,section,text FROM blocks WHERE page=? ORDER BY ordinal", (page,)):
                    print(json.dumps(dict(block), ensure_ascii=False))
    finally:
        index.close()


if __name__ == "__main__":
    main()
