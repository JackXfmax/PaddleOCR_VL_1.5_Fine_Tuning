#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from datetime import datetime

import paddle
from PIL import Image
from paddleformers.generation import GenerationConfig
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor

DEFAULT_MODEL_PATH = "/home/xufei/PaddleOCR-VL-1.5"
DEFAULT_PROMPT = "只识别图片中的中文文字，按阅读顺序输出；忽略藏文、英文和其他语言；如果没有中文则输出空字符串。"
NO_CHINESE_STRINGS = {
    "",
    "无",
    "无中文",
    "没有中文",
    "图片中无中文",
    "未识别到中文",
    "空",
    "none",
    "null",
    "n/a",
}
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")



def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", file=sys.stderr, flush=True)



def load_components(model_path: str):
    log(f"loading model: {model_path}")
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_path,
        convert_from_hf=True,
    ).eval()

    try:
        model.config._attn_implementation = "flashmask"
    except Exception:
        pass

    try:
        model.visual.config._attn_implementation = "flashmask"
    except Exception:
        pass

    processor = AutoProcessor.from_pretrained(model_path)
    generation_config = GenerationConfig(
        do_sample=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True,
        repetition_penalty=1.1,
    )
    return model, processor, generation_config



def infer_one(model, processor, generation_config, image_path: str, prompt: str, max_new_tokens: int) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pd",
        )
    with paddle.no_grad():
        generated_ids = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )
        generated_ids = generated_ids[0].tolist()[0]
        pred = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return pred



def normalize_prediction(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = LOC_TOKEN_RE.sub("", text)
    text = SPECIAL_TOKEN_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    lowered = text.lower()
    if lowered in NO_CHINESE_STRINGS:
        return ""

    kept_lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = NON_CHINESE_TEXT_RE.sub("", line)
        line = re.sub(r"\s+", " ", line).strip()
        if CJK_RE.search(line):
            kept_lines.append(line)

    return "\n".join(kept_lines).strip()





def find_assistant_content(record: dict) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "assistant":
            return (msg.get("content") or "").strip()
    raise ValueError("assistant message not found")



def merge_candidate(original_assistant: str, auto_chinese: str) -> str:
    original_assistant = (original_assistant or "").strip()
    auto_chinese = normalize_prediction(auto_chinese)
    if not auto_chinese:
        return original_assistant
    if auto_chinese in original_assistant:
        return original_assistant
    if not original_assistant:
        return auto_chinese
    return original_assistant + "\n" + auto_chinese



def build_review_record(record: dict, auto_chinese_raw: str, prompt: str, model_path: str) -> dict:
    original_assistant = find_assistant_content(record)
    auto_chinese = normalize_prediction(auto_chinese_raw)
    out_record = dict(record)
    out_record["auto_chinese_prompt"] = prompt
    out_record["auto_chinese_model"] = model_path
    out_record["auto_chinese_raw"] = (auto_chinese_raw or "").strip()
    out_record["auto_chinese_label"] = auto_chinese
    out_record["has_auto_chinese"] = bool(auto_chinese)
    out_record["merged_label_candidate"] = merge_candidate(original_assistant, auto_chinese)
    out_record["review_status"] = "pending"
    return out_record




def main() -> None:
    parser = argparse.ArgumentParser(description="Supplement original OCR jsonl with auto-recognized Chinese labels while preserving Tibetan annotations.")
    parser.add_argument("--input", required=True, help="Input jsonl path")
    parser.add_argument("--output", required=True, help="Output review jsonl path")
    parser.add_argument("--image_root", required=True, help="Directory containing images")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Model path for Chinese OCR supplementation")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used to extract Chinese only")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max generated tokens")
    parser.add_argument("--error_log", default="", help="Optional jsonl file to record skipped/error samples")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N lines, 0 means all")
    parser.add_argument("--start", type=int, default=0, help="Start from line offset")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"input not found: {args.input}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.error_log:
        os.makedirs(os.path.dirname(args.error_log) or ".", exist_ok=True)

    model, processor, generation_config = load_components(args.model_path)

    processed = 0
    skipped = 0
    added = 0
    error_fout = open(args.error_log, "w", encoding="utf-8") if args.error_log else None
    try:
        with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
            for idx, line in enumerate(fin):
                if idx < args.start:
                    continue
                if args.limit and processed >= args.limit:
                    break

                line = line.strip()
                if not line:
                    continue

                image_name = None
                try:
                    record = json.loads(line)
                    image_name = record["images"][0]
                    image_path = os.path.join(args.image_root, image_name)
                    if not os.path.isfile(image_path):
                        raise FileNotFoundError(f"image not found: {image_path}")

                    prediction_raw = infer_one(
                        model=model,
                        processor=processor,
                        generation_config=generation_config,
                        image_path=image_path,
                        prompt=args.prompt,
                        max_new_tokens=args.max_new_tokens,
                    )
                    prediction = normalize_prediction(prediction_raw)
                    if prediction:
                        added += 1

                    out_record = build_review_record(
                        record=record,
                        auto_chinese_raw=prediction_raw,
                        prompt=args.prompt,
                        model_path=args.model_path,
                    )


                    fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    fout.flush()
                    processed += 1

                    if processed % 20 == 0:
                        os.fsync(fout.fileno())
                        log(f"processed {processed} samples; added_chinese {added}; skipped {skipped}; latest image: {image_name}")
                except Exception as exc:
                    skipped += 1
                    error_payload = {
                        "idx": idx,
                        "image": image_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    log(f"skip idx={idx} image={image_name or 'UNKNOWN'} error={type(exc).__name__}: {exc}")
                    if error_fout is not None:
                        error_fout.write(json.dumps(error_payload, ensure_ascii=False) + "\n")
                        error_fout.flush()
                    continue
    finally:
        if error_fout is not None:
            error_fout.close()

    log(f"done. wrote {processed} samples to {args.output}; added_chinese {added}; skipped {skipped}")


if __name__ == "__main__":
    main()
