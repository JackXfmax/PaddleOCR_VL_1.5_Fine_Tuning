#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime

import paddle
from PIL import Image
from paddleformers.generation import GenerationConfig
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor

DEFAULT_MODEL_PATH = "/home/xufei/PaddleOCR-VL-1.5"
DEFAULT_PROMPT = "\u8bc6\u522b\u56fe\u7247\u4e2d\u7684\u6240\u6709\u6587\u5b57\uff0c\u6309\u9605\u8bfb\u987a\u5e8f\u8f93\u51fa\uff0c\u4e0d\u8981\u91cd\u590d\u3002"




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



def build_output_record(image_name: str, prompt: str, prediction: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"<image>{prompt}"},
            {"role": "assistant", "content": prediction},
        ],
        "images": [image_name],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Tibetan-only OCR jsonl into multilingual full-transcription jsonl.")
    parser.add_argument("--input", required=True, help="Input jsonl path")
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument("--image_root", required=True, help="Directory containing images")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="Base model path for auto transcription")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for multilingual transcription")
    parser.add_argument("--max_new_tokens", type=int, default=96, help="Max generated tokens")
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

                    prediction = infer_one(
                        model=model,
                        processor=processor,
                        generation_config=generation_config,
                        image_path=image_path,
                        prompt=args.prompt,
                        max_new_tokens=args.max_new_tokens,
                    )
                    out_record = build_output_record(image_name=image_name, prompt=args.prompt, prediction=prediction)
                    fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    fout.flush()
                    processed += 1

                    if processed % 20 == 0:
                        os.fsync(fout.fileno())
                        log(f"processed {processed} samples; skipped {skipped}; latest image: {image_name}")
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

    log(f"done. wrote {processed} samples to {args.output}; skipped {skipped}")



if __name__ == "__main__":
    main()
