#!/usr/bin/env bash
set -euo pipefail

cd /home/xufei/tibetan_ocr_lora

export CUDA_VISIBLE_DEVICES=1

PYTHON_BIN="/home/xufei/miniconda3/envs/ocr_vlm/bin/python"
SCRIPT_PATH="/home/xufei/tibetan_ocr_lora/supplement_chinese_labels.py"
DATA_ROOT="/home/xufei/natural_scene"

TRAIN_OUT="$DATA_ROOT/train_with_auto_chinese_review.jsonl"
TEST_OUT="$DATA_ROOT/test_with_auto_chinese_review.jsonl"
TRAIN_ERR="/home/xufei/tibetan_ocr_lora/train_with_auto_chinese_review_errors.jsonl"
TEST_ERR="/home/xufei/tibetan_ocr_lora/test_with_auto_chinese_review_errors.jsonl"
TRAIN_LOG="/home/xufei/tibetan_ocr_lora/convert_train_chinese_review.log"
TEST_LOG="/home/xufei/tibetan_ocr_lora/convert_test_chinese_review.log"

rm -f "$TRAIN_OUT" "$TEST_OUT" "$TRAIN_ERR" "$TEST_ERR" "$TRAIN_LOG" "$TEST_LOG"

"$PYTHON_BIN" -u "$SCRIPT_PATH" \
  --input "$DATA_ROOT/train.jsonl" \
  --output "$TRAIN_OUT" \
  --image_root "$DATA_ROOT" \
  --error_log "$TRAIN_ERR" \
  > "$TRAIN_LOG" 2>&1

"$PYTHON_BIN" -u "$SCRIPT_PATH" \
  --input "$DATA_ROOT/test.jsonl" \
  --output "$TEST_OUT" \
  --image_root "$DATA_ROOT" \
  --error_log "$TEST_ERR" \
  > "$TEST_LOG" 2>&1
