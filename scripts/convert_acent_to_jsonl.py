#!/usr/bin/env python3
"""
将 acent 藏文古籍数据集转换为 PaddleOCR-VL 训练用的 JSONL 格式

acent 目录结构:
  acent/
    lines/            # 藏文古籍图片（.png/.jpg 等）
    transcriptions/   # 对应的 .txt 标签

输出格式（PaddleOCR-VL-1.5）:
  {"messages": [{"role": "user", "content": "<image>OCR"}, {"role": "assistant", "content": "..."}], "images": ["..."]}
"""

import os
import json
import argparse

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def convert_acent_to_jsonl(acent_root, image_root=None, output_path=None, split_ratio=0.9, seed=42):
    """
    Args:
        acent_root:    acent 根目录（包含 lines/ 和 transcriptions/）
        image_root:    图片绝对路径前缀（用于 JSONL 中的 images 字段）。默认等于 acent_root。
        output_path:   输出 JSONL 路径。默认 acent_root/acent_paddle.jsonl
        split_ratio:   训练集比例，默认 0.9
        seed:          随机种子
    """
    lines_dir = os.path.join(acent_root, "lines")
    trans_dir = os.path.join(acent_root, "transcriptions")

    if not os.path.isdir(lines_dir):
        print(f"错误: lines 目录不存在 -> {lines_dir}")
        return
    if not os.path.isdir(trans_dir):
        print(f"错误: transcriptions 目录不存在 -> {trans_dir}")
        return

    if image_root is None:
        image_root = acent_root

    # 收集配对数据
    pairs = []
    for fname in sorted(os.listdir(lines_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTS:
            continue

        # 查找对应 txt（优先同名 .txt）
        txt_path = os.path.join(trans_dir, stem + ".txt")
        if not os.path.isfile(txt_path):
            # 尝试大小写变体
            for variant in [stem + ".TXT", stem + ".Txt", stem.capitalize() + ".txt"]:
                candidate = os.path.join(trans_dir, variant)
                if os.path.isfile(candidate):
                    txt_path = candidate
                    break

        if not os.path.isfile(txt_path):
            print(f"警告: 未找到 {fname} 对应的标签，跳过")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"警告: {fname} 的标签为空，跳过")
            continue

        image_rel = os.path.join("lines", fname)
        image_abs = os.path.join(image_root, image_rel)

        pairs.append({
            "messages": [
                {"role": "user", "content": "<image>OCR"},
                {"role": "assistant", "content": text}
            ],
            "images": [image_abs]
        })

    print(f"共收集 {len(pairs)} 条有效配对")

    if len(pairs) == 0:
        print("没有可用数据，退出")
        return

    # 划分训练/测试集
    import random
    random.seed(seed)
    indices = list(range(len(pairs)))
    random.shuffle(indices)

    n_train = int(len(pairs) * split_ratio)
    train_indices = set(indices[:n_train])
    test_indices = set(indices[n_train:])

    train_data = [pairs[i] for i in sorted(train_indices)]
    test_data = [pairs[i] for i in sorted(test_indices)]

    # 确定输出路径
    if output_path is None:
        output_path = os.path.join(acent_root, "acent_paddle.jsonl")

    train_path = output_path.replace(".jsonl", "_train.jsonl")
    test_path = output_path.replace(".jsonl", "_test.jsonl")

    def write_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {path}: {len(data)} 条")

    print(f"\n划分: 训练 {len(train_data)} / 测试 {len(test_data)} (ratio={split_ratio}, seed={seed})")
    print("写入文件:")
    write_jsonl(train_data, train_path)
    write_jsonl(test_data, test_path)
    # 同时写入全量文件
    write_jsonl(pairs, output_path)

    print(f"\n完成！全量: {output_path}")
    print(f"训练: {train_path}")
    print(f"测试: {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="acent 藏文古籍数据 → PaddleOCR-VL JSONL")
    parser.add_argument("--acent_root", type=str, default="/home/xufei/acent",
                        help="acent 根目录路径")
    parser.add_argument("--image_root", type=str, default=None,
                        help="图片绝对路径前缀（默认等于 acent_root）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSONL 路径（默认 acent_root/acent_paddle.jsonl）")
    parser.add_argument("--split_ratio", type=float, default=0.9,
                        help="训练集比例，默认 0.9")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，默认 42")
    args = parser.parse_args()

    convert_acent_to_jsonl(
        acent_root=args.acent_root,
        image_root=args.image_root,
        output_path=args.output,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )
