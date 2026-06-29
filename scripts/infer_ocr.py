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

DEFAULT_MODEL_PATH = "/home/xufei/tibetan_ocr_lora/export"
DEFAULT_PROMPT = "识别图片中的藏文"


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def load_components(model_path):
    log(f"加载模型: {model_path}")
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
    )
    return model, processor, generation_config


def infer_one(model, processor, generation_config, image_path, prompt, max_new_tokens):
    image = Image.open(image_path).convert("RGB")
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


def main():
    parser = argparse.ArgumentParser(description="对一张或多张图片执行 OCR 推理")
    parser.add_argument("images", nargs="+", help="一张或多张图片的绝对路径")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="模型目录，默认使用微调导出模型")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="推理提示词")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="生成最大 token 数")
    parser.add_argument("--save", default="", help="可选：将结果保存为 JSON 文件")
    parser.add_argument("--json", action="store_true", help="直接以 JSON 打印结果")
    args = parser.parse_args()

    missing = [path for path in args.images if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("以下图片不存在: " + ", ".join(missing))

    model, processor, generation_config = load_components(args.model_path)

    results = []
    for image_path in args.images:
        log(f"开始推理: {image_path}")
        pred = infer_one(
            model=model,
            processor=processor,
            generation_config=generation_config,
            image_path=image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        results.append({
            "image": image_path,
            "prompt": args.prompt,
            "prediction": pred,
        })

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log(f"结果已保存到: {args.save}")

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    for idx, item in enumerate(results, 1):
        print(f"===== 结果 {idx} =====")
        print(f"图片: {item['image']}")
        print(f"提示词: {item['prompt']}")
        print("OCR:")
        print(item["prediction"])
        print()


if __name__ == "__main__":
    main()
