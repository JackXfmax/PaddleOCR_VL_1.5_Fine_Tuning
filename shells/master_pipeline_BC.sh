#!/bin/bash
# master_pipeline_BC.sh
# 全流程：中文补充 -> 构造B/C训练集 -> 训练B/C -> 导出 -> 评测 -> 汇总
# 训练B(joint_full)用GPU0, 训练C(multilingual_full基座重训)用GPU1

set -e
PYTHON=/home/xufei/miniconda3/envs/ocr_vlm/bin/python
LORA_DIR=/home/xufei/tibetan_ocr_lora
DATA=/home/xufei/natural_scene
BASE_MODEL=/home/xufei/PaddleOCR-VL-1.5
MODEL_A=/home/xufei/tibetan_ocr_lora/export
FORMERS=/home/xufei/PaddleFormers-develop

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $LORA_DIR/master_BC.log; }

source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate ocr_vlm

# =========================================================
# STEP 1: 用GPU0生成 train 中文补充（约1800条）
# =========================================================
log "STEP1: generating train chinese supplement..."
CUDA_VISIBLE_DEVICES=0 $PYTHON -u $LORA_DIR/supplement_chinese_labels.py \
    --input $DATA/train.jsonl \
    --output $DATA/train_with_auto_chinese_review.jsonl \
    --image_root $DATA \
    --error_log $LORA_DIR/supplement_train_errors.jsonl \
    2>>$LORA_DIR/supplement_train.log
log "STEP1 train done. lines=$(wc -l < $DATA/train_with_auto_chinese_review.jsonl)"

# =========================================================
# STEP 2: 用GPU0生成 test 中文补充（约200条）
# =========================================================
log "STEP2: generating test chinese supplement..."
CUDA_VISIBLE_DEVICES=0 $PYTHON -u $LORA_DIR/supplement_chinese_labels.py \
    --input $DATA/test.jsonl \
    --output $DATA/test_with_auto_chinese_review.jsonl \
    --image_root $DATA \
    --error_log $LORA_DIR/supplement_test_errors.jsonl \
    2>>$LORA_DIR/supplement_test.log
log "STEP2 test done. lines=$(wc -l < $DATA/test_with_auto_chinese_review.jsonl)"

# =========================================================
# STEP 3: 构造B/C实验所需训练集
# =========================================================
log "STEP3: building experiment datasets..."
$PYTHON $LORA_DIR/build_multilingual_experiment_sets.py \
    --original $DATA/train.jsonl \
    --review $DATA/train_with_auto_chinese_review.jsonl \
    --multilingual_full $DATA/train_multilingual_full.jsonl \
    --multilingual_zhpos $DATA/train_multilingual_zhpos.jsonl \
    --joint_full $DATA/train_joint_full.jsonl \
    --joint_zhpos $DATA/train_joint_zhpos.jsonl \
    2>>$LORA_DIR/build_datasets.log

# 同样构造test评测集（多语言版）
$PYTHON $LORA_DIR/build_multilingual_experiment_sets.py \
    --original $DATA/test.jsonl \
    --review $DATA/test_with_auto_chinese_review.jsonl \
    --multilingual_full $DATA/test_multilingual_full.jsonl \
    --multilingual_zhpos $DATA/test_multilingual_zhpos.jsonl \
    --joint_full /tmp/test_joint_full_discard.jsonl \
    --joint_zhpos /tmp/test_joint_zhpos_discard.jsonl \
    2>>$LORA_DIR/build_datasets.log

log "STEP3 done."
for f in train_multilingual_full train_multilingual_zhpos train_joint_full train_joint_zhpos \
         test_multilingual_full; do
    log "  $f: $(wc -l < $DATA/$f.jsonl) lines"
done

# =========================================================
# STEP 4A: 写实验B配置（joint_full，从基座LoRA微调，GPU0）
# =========================================================
cat > /home/xufei/exp_B_config.yaml << 'YAML'
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: /home/xufei/natural_scene/train_joint_full.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: /home/xufei/natural_scene/test_multilingual_full.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 4096
padding_free: True
truncate_packing: False
dataloader_num_workers: 4
mix_strategy: concat
template_backend: custom
template: paddleocr_vl_v15
custom_register_path: /home/xufei/paddleocr_vl_v15_template.py

### model
model_name_or_path: /home/xufei/PaddleOCR-VL-1.5
_attn_implementation: flashmask
lora: true
lora_rank: 8

### finetuning
stage: VL-SFT
fine_tuning: lora
seed: 42
do_train: true
do_eval: true
per_device_eval_batch_size: 4
per_device_train_batch_size: 4
num_train_epochs: 3
max_steps: -1
eval_steps: 300
evaluation_strategy: steps
save_steps: 300
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 4
logging_dir: /home/xufei/tibetan_ocr_lora/exp_B/vdl_log/
output_dir: /home/xufei/tibetan_ocr_lora/exp_B/

disable_tqdm: true
eval_accumulation_steps: 16

lr_scheduler_type: cosine
warmup_ratio: 0.1
learning_rate: 5.0e-4
min_lr: 5.0e-5

weight_decay: 0.1
adam_epsilon: 1.0e-8
adam_beta1: 0.9
adam_beta2: 0.95

tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
sharding: stage1
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
fp16_opt_level: O2

unified_checkpoint: False
save_checkpoint_format: "flex_checkpoint"
load_checkpoint_format: "flex_checkpoint"
YAML
log "exp_B_config.yaml written"

# =========================================================
# STEP 4B: 写实验C配置（multilingual_full，从基座LoRA微调，GPU1）
# =========================================================
cat > /home/xufei/exp_C_config.yaml << 'YAML'
### data
train_dataset_type: messages
eval_dataset_type: messages
train_dataset_path: /home/xufei/natural_scene/train_multilingual_full.jsonl
train_dataset_prob: "1.0"
eval_dataset_path: /home/xufei/natural_scene/test_multilingual_full.jsonl
eval_dataset_prob: "1.0"
max_seq_len: 4096
padding_free: True
truncate_packing: False
dataloader_num_workers: 4
mix_strategy: concat
template_backend: custom
template: paddleocr_vl_v15
custom_register_path: /home/xufei/paddleocr_vl_v15_template.py

### model
model_name_or_path: /home/xufei/PaddleOCR-VL-1.5
_attn_implementation: flashmask
lora: true
lora_rank: 8

### finetuning
stage: VL-SFT
fine_tuning: lora
seed: 42
do_train: true
do_eval: true
per_device_eval_batch_size: 4
per_device_train_batch_size: 4
num_train_epochs: 3
max_steps: -1
eval_steps: 300
evaluation_strategy: steps
save_steps: 300
save_strategy: steps
logging_steps: 1
gradient_accumulation_steps: 4
logging_dir: /home/xufei/tibetan_ocr_lora/exp_C/vdl_log/
output_dir: /home/xufei/tibetan_ocr_lora/exp_C/

disable_tqdm: true
eval_accumulation_steps: 16

lr_scheduler_type: cosine
warmup_ratio: 0.1
learning_rate: 5.0e-4
min_lr: 5.0e-5

weight_decay: 0.1
adam_epsilon: 1.0e-8
adam_beta1: 0.9
adam_beta2: 0.95

tensor_model_parallel_size: 1
pipeline_model_parallel_size: 1
sharding: stage1
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
fp16_opt_level: O2

unified_checkpoint: False
save_checkpoint_format: "flex_checkpoint"
load_checkpoint_format: "flex_checkpoint"
YAML
log "exp_C_config.yaml written"

# =========================================================
# STEP 5: 并行训练B(GPU0)和C(GPU1)
# =========================================================
mkdir -p $LORA_DIR/exp_B $LORA_DIR/exp_C
log "STEP5: launching training B on GPU0 and C on GPU1 in parallel..."

cd $FORMERS
CUDA_VISIBLE_DEVICES=0 python -m paddleformers.cli.cli train /home/xufei/exp_B_config.yaml \
    > $LORA_DIR/exp_B/train.log 2>&1 &
PID_B=$!
log "  exp_B PID=$PID_B"

CUDA_VISIBLE_DEVICES=1 python -m paddleformers.cli.cli train /home/xufei/exp_C_config.yaml \
    > $LORA_DIR/exp_C/train.log 2>&1 &
PID_C=$!
log "  exp_C PID=$PID_C"

# 等两个训练都完成
wait $PID_B
log "STEP5: exp_B training done (PID $PID_B)"
wait $PID_C
log "STEP5: exp_C training done (PID $PID_C)"

# =========================================================
# STEP 6: 导出B和C（用 paddleformers-cli export，与原始A一致）
# =========================================================
cat > /home/xufei/exp_B_export_config.yaml << YAML
model_name_or_path: /home/xufei/PaddleOCR-VL-1.5
output_dir: /home/xufei/tibetan_ocr_lora/exp_B
lora: true
stage: VL-SFT
fine_tuning: lora
YAML

cat > /home/xufei/exp_C_export_config.yaml << YAML
model_name_or_path: /home/xufei/PaddleOCR-VL-1.5
output_dir: /home/xufei/tibetan_ocr_lora/exp_C
lora: true
stage: VL-SFT
fine_tuning: lora
YAML

log "STEP6: exporting exp_B..."
cd $FORMERS
CUDA_VISIBLE_DEVICES=0 python -m paddleformers.cli.cli export /home/xufei/exp_B_export_config.yaml \
    > $LORA_DIR/export_B.log 2>&1
log "STEP6: export_B done"

log "STEP6: exporting exp_C..."
CUDA_VISIBLE_DEVICES=1 python -m paddleformers.cli.cli export /home/xufei/exp_C_export_config.yaml \
    > $LORA_DIR/export_C.log 2>&1
log "STEP6: export_C done"

# 找到真实 export 目录（export 子目录里）
EXPORT_B=$LORA_DIR/exp_B/export
EXPORT_C=$LORA_DIR/exp_C/export

# =========================================================
# STEP 7: 对所有模型评测（藏文test + 多语言test）
# =========================================================
log "STEP7: evaluating all models..."

# 7a: 模型A(已有) -- 藏文测试集（已有eval_finetuned_200.json，不重跑）
# 7b: 基座（已有eval_base_200.json，不重跑）

# 7c: exp_B 藏文测试集
CUDA_VISIBLE_DEVICES=0 $PYTHON $LORA_DIR/full_eval.py \
    --model_path $EXPORT_B \
    --data_path $DATA/test.jsonl \
    --image_dir $DATA \
    --output_path $LORA_DIR/eval_B_tibetan.json \
    > $LORA_DIR/eval_B_tibetan.log 2>&1 &
PID_EVALB=$!

# 7d: exp_C 藏文测试集
CUDA_VISIBLE_DEVICES=1 $PYTHON $LORA_DIR/full_eval.py \
    --model_path $EXPORT_C \
    --data_path $DATA/test.jsonl \
    --image_dir $DATA \
    --output_path $LORA_DIR/eval_C_tibetan.json \
    > $LORA_DIR/eval_C_tibetan.log 2>&1 &
PID_EVALC=$!
wait $PID_EVALB $PID_EVALC
log "STEP7a/b: tibetan eval done"

# 7e: 基座多语言评测
CUDA_VISIBLE_DEVICES=0 $PYTHON $LORA_DIR/full_eval.py \
    --model_path $BASE_MODEL \
    --data_path $DATA/test_multilingual_full.jsonl \
    --image_dir $DATA \
    --output_path $LORA_DIR/eval_base_multilingual.json \
    > $LORA_DIR/eval_base_multilingual.log 2>&1 &
PID_BASE_ML=$!

# 7f: 模型A 多语言评测
CUDA_VISIBLE_DEVICES=1 $PYTHON $LORA_DIR/full_eval.py \
    --model_path $MODEL_A \
    --data_path $DATA/test_multilingual_full.jsonl \
    --image_dir $DATA \
    --output_path $LORA_DIR/eval_A_multilingual.json \
    > $LORA_DIR/eval_A_multilingual.log 2>&1 &
PID_A_ML=$!
wait $PID_BASE_ML $PID_A_ML
log "STEP7c/d: multilingual base+A eval done"

# 7g: exp_B 多语言评测
CUDA_VISIBLE_DEVICES=0 $PYTHON $LORA_DIR/full_eval.py \
    --model_path $EXPORT_B \
    --data_path $DATA/test_multilingual_full.jsonl \
    --image_dir $DATA \
    --output_path $LORA_DIR/eval_B_multilingual.json \
    > $LORA_DIR/eval_B_multilingual.log 2>&1 &
PID_EVALB_ML=$!

# 7h: exp_C 多语言评测
CUDA_VISIBLE_DEVICES=1 $PYTHON $LORA_DIR/full_eval.py \
    --model_path $EXPORT_C \
    --data_path $DATA/test_multilingual_full.jsonl \
    --image_dir $DATA \
    --output_path $LORA_DIR/eval_C_multilingual.json \
    > $LORA_DIR/eval_C_multilingual.log 2>&1 &
PID_EVALC_ML=$!
wait $PID_EVALB_ML $PID_EVALC_ML
log "STEP7e/f: multilingual B+C eval done"

# =========================================================
# STEP 8: 生成汇总结果JSON
# =========================================================
log "STEP8: generating summary..."
$PYTHON - << 'PYEOF'
import json, os

def load_summary(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding='utf-8'))
    return {
        "model_path": d.get("model_path", ""),
        "samples": d.get("samples", 0),
        "avg_sim": round(d.get("avg_normalized_similarity", 0), 4),
        "exact_match_rate": round(d.get("exact_match_rate", 0), 4),
        "exact_match_count": d.get("exact_match_count", 0),
    }

BASE_LORA = "/home/xufei/tibetan_ocr_lora"

summary = {
    "tibetan_test": {
        "base":    load_summary(f"{BASE_LORA}/eval_base_200.json"),
        "A_tibetan_lora": load_summary(f"{BASE_LORA}/eval_finetuned_200.json"),
        "B_joint_lora":   load_summary(f"{BASE_LORA}/eval_B_tibetan.json"),
        "C_multilingual_lora": load_summary(f"{BASE_LORA}/eval_C_tibetan.json"),
    },
    "multilingual_test": {
        "base":    load_summary(f"{BASE_LORA}/eval_base_multilingual.json"),
        "A_tibetan_lora": load_summary(f"{BASE_LORA}/eval_A_multilingual.json"),
        "B_joint_lora":   load_summary(f"{BASE_LORA}/eval_B_multilingual.json"),
        "C_multilingual_lora": load_summary(f"{BASE_LORA}/eval_C_multilingual.json"),
    }
}

out = f"{BASE_LORA}/ablation_summary.json"
with open(out, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"Summary written to {out}")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PYEOF
log "STEP8 done. Pipeline complete."
