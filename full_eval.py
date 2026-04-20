#!/usr/bin/env python3
"""
PaddleOCR-VL 全量评测脚本
计算 NED (Normalized Edit Distance) 和 Exact Match Rate
"""

import json
import argparse
import sys
import os
import time
from datetime import datetime
from difflib import SequenceMatcher


def ned_similarity(pred: str, gt: str) -> float:
    """计算归一化编辑距离相似度（SequenceMatcher ratio）"""
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    return SequenceMatcher(None, pred, gt).ratio()


def evaluate(args):
    """主评测函数"""
    # 加载测试数据
    test_data = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Loaded {len(test_data)} test samples from {args.data_path}")

    # 导入 PaddleNLP / PaddleFormers 推理组件
    from paddlenlp.taskflow import Taskflow

    # 初始化模型
    model_kwargs = {}
    if args.lora_path:
        model_kwargs["lora_path"] = args.lora_path

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Loading model from {args.model_path} ...")
    ocr_taskflow = Taskflow(
        "document_intelligence",
        model=args.model_path,
        **model_kwargs,
    )

    # 逐条推理
    results = []
    similarities = []
    exact_matches = 0

    for i, item in enumerate(test_data):
        # 提取图片路径和 ground truth
        image_path = item["images"][0]
        if args.image_dir and not os.path.isabs(image_path):
            image_path = os.path.join(args.image_dir, image_path)

        # ground truth: assistant 最后一条 content
        gt = ""
        for msg in reversed(item["messages"]):
            if msg["role"] == "assistant":
                gt = msg["content"]
                break

        try:
            # 推理
            result = ocr_taskflow({"doc": image_path})
            # 提取预测文本
            if isinstance(result, list) and len(result) > 0:
                pred = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
            elif isinstance(result, dict):
                pred = result.get("text", str(result))
            elif isinstance(result, str):
                pred = result
            else:
                pred = str(result)
        except Exception as e:
            pred = f"[ERROR] {str(e)}"

        sim = ned_similarity(pred, gt)
        is_exact = (pred == gt)
        if is_exact:
            exact_matches += 1

        similarities.append(sim)
        results.append({
            "index": i,
            "image": item["images"][0],
            "ground_truth": gt,
            "prediction": pred,
            "similarity": round(sim, 6),
            "exact_match": is_exact
        })

        if (i + 1) % args.log_every == 0 or (i + 1) == len(test_data):
            avg_sim = sum(similarities) / len(similarities)
            em_rate = exact_matches / len(similarities)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Processed {len(similarities)} samples | "
                  f"avg_similarity={avg_sim:.6f} | exact_match_rate={em_rate:.4f}")

    # 汇总
    avg_ned = sum(similarities) / len(similarities)
    avg_dist = 1.0 - avg_ned
    em_rate = exact_matches / len(test_data)

    summary = {
        "samples": len(test_data),
        "avg_normalized_distance": avg_dist,
        "avg_normalized_similarity": avg_ned,
        "exact_match_rate": em_rate,
        "exact_match_count": exact_matches,
    }

    # 保存结果
    output = {
        "summary": summary,
        "details": results,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Saved evaluation summary to {args.output_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaddleOCR-VL Full Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base model or exported model")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter weights (optional)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test JSONL file")
    parser.add_argument("--image_dir", type=str, default="/",
                        help="Root directory for images (default: /)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save evaluation results JSON")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Print progress every N samples")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens for generation")

    args = parser.parse_args()
    evaluate(args)
