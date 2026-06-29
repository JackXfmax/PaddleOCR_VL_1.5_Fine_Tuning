#!/usr/bin/env python3
import argparse
import json
import os
from typing import Iterable

MULTILINGUAL_PROMPT = "<image>逐字转写图片中实际出现的所有文字，按阅读顺序输出，保持原语言，不要翻译，不要解释，不要补充图片中不存在的内容。"


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)



def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count



def get_assistant(record: dict) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            return (msg.get("content") or "").strip()
    raise ValueError("assistant message not found")



def make_multilingual_record(record: dict, assistant_text: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": MULTILINGUAL_PROMPT},
            {"role": "assistant", "content": (assistant_text or "").strip()},
        ],
        "images": list(record.get("images", [])),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Build multilingual/joint experiment datasets from Chinese-supplement review jsonl.")
    parser.add_argument("--original", required=True, help="Original tibetan-only jsonl")
    parser.add_argument("--review", required=True, help="Review jsonl produced by supplement_chinese_labels.py")
    parser.add_argument("--multilingual_full", required=True, help="Output multilingual dataset for all samples")
    parser.add_argument("--multilingual_zhpos", required=True, help="Output multilingual dataset for samples with non-empty auto Chinese only")
    parser.add_argument("--joint_full", required=True, help="Output joint dataset: original + multilingual_full")
    parser.add_argument("--joint_zhpos", required=True, help="Output joint dataset: original + multilingual_zhpos")
    args = parser.parse_args()

    original_rows = list(iter_jsonl(args.original))
    review_rows = list(iter_jsonl(args.review))
    if len(original_rows) != len(review_rows):
        raise ValueError(f"row count mismatch: original={len(original_rows)} review={len(review_rows)}")

    multilingual_full_rows = []
    multilingual_zhpos_rows = []

    for original, review in zip(original_rows, review_rows):
        merged = (review.get("merged_label_candidate") or "").strip()
        if not merged:
            merged = get_assistant(original)
        multilingual_row = make_multilingual_record(review, merged)
        multilingual_full_rows.append(multilingual_row)
        if review.get("has_auto_chinese"):
            multilingual_zhpos_rows.append(multilingual_row)

    full_count = write_jsonl(args.multilingual_full, multilingual_full_rows)
    zhpos_count = write_jsonl(args.multilingual_zhpos, multilingual_zhpos_rows)
    joint_full_count = write_jsonl(args.joint_full, list(original_rows) + multilingual_full_rows)
    joint_zhpos_count = write_jsonl(args.joint_zhpos, list(original_rows) + multilingual_zhpos_rows)

    summary = {
        "original": len(original_rows),
        "review": len(review_rows),
        "multilingual_full": full_count,
        "multilingual_zhpos": zhpos_count,
        "joint_full": joint_full_count,
        "joint_zhpos": joint_zhpos_count,
        "multilingual_prompt": MULTILINGUAL_PROMPT,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
