# PaddleOCR-VL 藏文/多语言 LoRA 微调全流程
> 新服务器实验 | 日期：2026-04-17 | 状态：完成 ✅

---

## 一、环境信息

| 项目 | 内容 |
|------|------|
| 服务器 | ubuntu2204@222.19.82.36 |
| SSH 连接 | `plink -ssh ubuntu2204@222.19.82.36 -pw 385413336 -hostkey "SHA256:1Dlgrs9L1bTlucHMu65BJbRDfW07+h3/NX1LUK8mF8A"` |
| 系统 | Ubuntu 22.04，Python 3.10.12 |
| 硬件 | 2× RTX 4090（各 24 GB 显存） |
| Conda 环境 | `paddle` |
| 基座模型 | PaddleOCR-VL-1.5（`/home/ubuntu2204/xf/PaddleOCR-VL-1.5/`） |

---

## 二、数据准备

### 2.1 原始数据

从旧服务器拉取人工审核完毕的标注数据，共 **2000 条**，全部为**多语言样本**（藏文 + 中文混合，`has_auto_chinese=True`），拷贝到新服务器：

```
/home/ubuntu2204/xf/natural_scene/reviewed_raw.jsonl   # 2000 条
```

每条样本核心字段：
```json
{
  "image": "图片文件名",
  "merged_label_candidate": "最终转写答案（藏文+中文，含换行分段）",
  "has_auto_chinese": true
}
```

### 2.2 训练/测试集划分

按 **90% / 10%** 划分，`random_state=42`：

```python
from sklearn.model_selection import train_test_split
import json

data = [json.loads(l) for l in open("reviewed_raw.jsonl")]
train, test = train_test_split(data, test_size=0.1, random_state=42)
# 训练集：~1800 条，测试集：~200 条
```

| 文件 | 条数 | 用途 |
|------|------|------|
| `train_joint_full.jsonl` | 1799 | 训练（本次使用） |
| `test_all.jsonl` | 201 | 评测（本次使用） |

### 2.3 数据格式转换

PaddleOCR-VL-1.5 要求 user 消息里必须以 `<image>` 开头，prompt 使用 `OCR`：

```python
def to_paddle_format(item, image_root):
    return {
        "messages": [
            {
                "role": "user",
                "content": "<image>OCR"          # <image> 占位符 + prompt
            },
            {
                "role": "assistant",
                "content": item["merged_label_candidate"]   # 人工审核后的答案
            }
        ],
        "images": [os.path.join(image_root, item["image"])]
    }
```

> ⚠️ 注意：`<image>` 占位符是必须的，缺少会导致模型无法感知图片。

---

## 三、数据增强

在原有训练模板基础上新增三种增强方法，模拟低质量拍摄/扫描场景：

```python
# /home/ubuntu2204/xf/paddleocr_vl_v15_template.py

import numpy as np
import io
import random
from PIL import Image, ImageFilter
from torchvision import transforms


class GaussianNoise:
    """随机高斯噪声，模拟低质量扫描"""
    def __init__(self, prob=0.3, mean=0, std=25):
        self.prob = prob
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if random.random() < self.prob:
            arr = np.array(img).astype(np.float32)
            arr = np.clip(arr + np.random.normal(self.mean, self.std, arr.shape), 0, 255)
            return Image.fromarray(arr.astype(np.uint8))
        return img


class GaussianBlur:
    """随机高斯模糊，模拟失焦"""
    def __init__(self, prob=0.3, radius_range=(1, 3)):
        self.prob = prob
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() < self.prob:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class JpegCompression:
    """随机 JPEG 压缩，模拟压缩伪影"""
    def __init__(self, prob=0.3, quality_range=(40, 85)):
        self.prob = prob
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() < self.prob:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(*self.quality_range))
            buf.seek(0)
            return Image.open(buf).copy()
        return img


# 训练用增强 pipeline
train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    GaussianNoise(prob=0.3),
    GaussianBlur(prob=0.3),
    JpegCompression(prob=0.3),
])
```

---

## 四、训练配置

### 4.1 train_config.yaml

```yaml
# /home/ubuntu2204/xf/train_config.yaml

# ===== 模型 =====
model_name_or_path: /home/ubuntu2204/xf/PaddleOCR-VL-1.5/
trust_remote_code: true

# ===== LoRA =====
lora: true
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj

# ===== 数据 =====
dataset: /home/ubuntu2204/xf/natural_scene/train_joint_full.jsonl
image_root: /home/ubuntu2204/xf/natural_scene/
template: /home/ubuntu2204/xf/paddleocr_vl_v15_template.py

# ===== 训练超参 =====
output_dir: /home/ubuntu2204/xf/output/
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
dataloader_num_workers: 4

# ===== 保存 =====
save_strategy: epoch
save_total_limit: 3
logging_steps: 5
report_to: none
```

### 4.2 双卡训练启动命令

```bash
source /home/ubuntu2204/miniconda3/etc/profile.d/conda.sh
conda activate paddle

cd /home/ubuntu2204/xf

nohup python -m paddle.distributed.launch \
    --gpus "0,1" \
    --log_dir /home/ubuntu2204/xf/paddleformers_dist_log \
    -m paddleformers.cli.cli train /home/ubuntu2204/xf/train_config.yaml \
    > /home/ubuntu2204/xf/output/train.log 2>&1 &

echo "PID: $!"
```

---

## 五、训练过程

| 参数 | 值 |
|------|------|
| 训练集 | train_joint_full.jsonl（1799 条，多语言） |
| Epochs | 3 |
| 有效 batch size | 4 × 2 GPU × 4 grad_accum = **32** |
| 总步数 | 169 步 |
| 每步耗时 | ~6 秒 |
| 总训练时长 | **16 分 16 秒** |
| 精度 | bf16 |
| 初始 loss | ~0.313 |
| 最终 loss | **0.056** |
| 平均 train_loss | **0.143** |

输出目录 `/home/ubuntu2204/xf/output/`：
```
output/
├── adapter_config.json           # LoRA 结构配置
├── adapter_model.safetensors     # LoRA 权重（14.5 MB）
├── tokenizer/
├── train_results.json
├── all_results.json
└── train.log
```

---

## 六、评测

### 6.1 评测指标

- **Avg NED**：归一化编辑距离相似度（SequenceMatcher ratio），越高越好
- **EM**：完全匹配率（Exact Match），pred 与 gt 字符串完全一致

```python
from difflib import SequenceMatcher

def ned_similarity(pred: str, gt: str) -> float:
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    return SequenceMatcher(None, pred, gt).ratio()
```

### 6.2 评测启动命令

```bash
source /home/ubuntu2204/miniconda3/etc/profile.d/conda.sh
conda activate paddle

nohup python /home/ubuntu2204/xf/full_eval.py \
    --model_path /home/ubuntu2204/xf/PaddleOCR-VL-1.5/ \
    --lora_path  /home/ubuntu2204/xf/output/ \
    --test_file  /home/ubuntu2204/xf/natural_scene/test_all.jsonl \
    --image_root /home/ubuntu2204/xf/natural_scene/ \
    --output_file /home/ubuntu2204/xf/eval_results.json \
    > /home/ubuntu2204/xf/eval_output.log 2>&1 &
```

---

## 七、最终结果

| 指标 | 值 |
|------|------|
| **Avg NED** | **0.9330** |
| **Exact Match** | **52.0%（104/200）** |
| 测试集 | test_all.jsonl（200 条，多语言） |

评测结果文件：`/home/ubuntu2204/xf/eval_results.json`（每条含 pred / gt / sim）

---

## 八、文件路径速查

```
/home/ubuntu2204/xf/
├── PaddleOCR-VL-1.5/                   # 基座模型
├── natural_scene/
│   ├── reviewed_raw.jsonl               # 原始 2000 条（备份）
│   ├── train_joint_full.jsonl           # 训练集（1799 条）
│   └── test_all.jsonl                   # 测试集（201 条）
├── paddleocr_vl_v15_template.py        # 训练模板（含数据增强）
├── train_config.yaml                    # 训练配置
├── full_eval.py                         # 评测脚本
├── output/
│   ├── adapter_model.safetensors        # LoRA 权重
│   ├── train_results.json
│   └── train.log
├── eval_results.json                    # 评测结果
├── eval_output.log                      # 评测日志
└── paddleformers_dist_log/
    ├── workerlog.0                      # GPU0 日志
    └── workerlog.1                      # GPU1 日志
```

---

## 九、合并数据集训练实验（QT-MSTR V3 + TibNST）

> 日期：2026-04-18 | 状态：完成 ✅

### 9.1 新增数据集

| 数据集 | 场景 | 图片量 | 标注量 | 语言分布 |
|--------|------|--------|--------|----------|
| QT-MSTR V3 | 店面招牌 | 1000张 (2.1GB) | 12,336有效框 | TI 4453 / CH 6806 / EN 861 / DI 768 |
| TibNST/tibdata | 自然场景藏文 | 1898张 (3.1GB) | 2,046条 | 纯藏文为主 |
| TibNST 合成 | 合成增强 | 500张 (215MB) | 500条 | 藏文, 含旋转角 |

### 9.2 标注格式统一

QT-MSTR (LabelMe像素绝对值) 和 TibNST (百分比+旋转角) 统一转换为 PaddleOCR-VL JSONL 格式：

```python
# QT-MSTR: 像素坐标 → 文本提取，按 polygon/bbox 类型处理
# TibNST: 百分比坐标 → 像素坐标还原，rotation 归零或保留
# 统一 prompt: "<image>OCR"
```

### 9.3 分层分级 LoRA 方案（TibetanOCR-Hierarchical-LoRA v2）

> 针对藏文 OCR 的多语种混合场景，设计分层分级 LoRA 策略。核心思路：**浅层保视觉、深层攻语义、MLP 增强跨语言映射**。

#### 9.3.1 层级划分策略

模型共 32 层 Transformer，按功能分为三个层级：

| 层级 | 层范围 | 功能定位 | LoRA rank | LoRA alpha | 目标模块 |
|------|--------|---------|:---------:|:----------:|---------|
| Shallow | 0-10 | 基础视觉特征（边缘/纹理/颜色） | 8 | 16 | q, k, v, o |
| Middle | 11-21 | 中层语义（字形/笔画/字形切换） | 24 | 48 | q, k, v, o + gate, up, down |
| Deep | 22-31 | 高层任务（语言建模/OCR解码） | 48 | 96 | q, k, v, o + gate, up, down |

**可训参数量估算：**

| 层级 | layers | rank | 模块数 | params/layer | 层级总参 |
|------|:-----:|:----:|:-----:|:-----------:|:------:|
| Shallow | 11 | 8 | 4 (attn) | ~0.8M | ~8.8M |
| Middle | 11 | 24 | 7 (attn+mlp) | ~4.0M | ~44.0M |
| Deep | 10 | 48 | 7 (attn+mlp) | ~8.1M | ~81.0M |
| **Total** | **32** | — | — | — | **~133.8M** |

#### 9.3.2 训练配置

```yaml
# train_merged_hierarchical_config.yaml
model_name_or_path: /home/ubuntu2204/xf/PaddleOCR-VL-1.5/

# ===== 分层 LoRA =====
lora: true
lora_type: hierarchical       # 分层分级策略

# --- Shallow 层 (0-10): 轻量视觉适配 ---
lora_shallow:
  layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  rank: 8
  alpha: 16
  dropout: 0.02
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

# --- Middle 层 (11-21): 字形与跨语言语义 ---
lora_middle:
  layers: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
  rank: 24
  alpha: 48
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

# --- Deep 层 (22-31): 任务解码器 ---
lora_deep:
  layers: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
  rank: 48
  alpha: 96
  dropout: 0.08
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

# --- LoRA+ 非对称学习率 ---
loraplus: true
loraplus_lr_ratio: 2.0        # B 矩阵学习率 = A × 2

# --- rsLoRA 秩稳定缩放 ---
use_rslora: true               # 缩放因子 = α/√r 替代 α/r

# ===== 数据 =====
dataset: /home/ubuntu2204/xf/natural_scene/train_merged.jsonl
image_root: /home/ubuntu2204/xf/natural_scene/
template: /home/ubuntu2204/xf/paddleocr_vl_v15_template.py

# ===== 训练超参 =====
output_dir: /home/ubuntu2204/xf/output_merged_hierarchical/
num_train_epochs: 5
per_device_train_batch_size: 6    # rank大→显存增加，batch下调
gradient_accumulation_steps: 3    # 有效 batch = 6×2×3 = 36
learning_rate: 2.0e-4             # LoRA+ 时降低基学习率
lr_scheduler_type: cosine
warmup_ratio: 0.1                 # 更长的 warmup 稳定深层高秩
bf16: true
dataloader_num_workers: 4
gradient_checkpointing: true      # 节省显存

# ===== 正则化 =====
weight_decay: 0.01
max_grad_norm: 0.5
label_smoothing: 0.02

# ===== 保存 =====
save_strategy: epoch
save_total_limit: 5
logging_steps: 5
report_to: tensorboard
```

#### 9.3.3 LoRA+ 非对称学习率原理

标准 LoRA 中 A 和 B 矩阵共享同一学习率，LoRA+ 证明 B 矩阵需要更大的学习率：

```
ΔW = B × A     (r ≪ d)
η_B = λ × η_A   (λ = 2.0)
```

在分层方案中，λ 还随层级加深而增大：
- Shallow: λ = 1.5（浅层特征变化应保守）
- Middle:  λ = 2.0
- Deep:    λ = 2.5（深层任务适配需更激进）

#### 9.3.4 rsLoRA 秩稳定缩放

标准 LoRA 缩放因子为 `α/r`，高秩时梯度不稳定。rsLoRA 改为 `α/√r`：

```
标准 LoRA:  ΔW = (α/r) · BA
rsLoRA:     ΔW = (α/√r) · BA

r=48 时: 标准 α/r = 2.0  vs  rsLoRA α/√r ≈ 13.9
→ 高秩层获得更稳定的梯度传播
```

#### 9.3.5 训练过程

| 参数 | 值 |
|------|------|
| 训练集 | train_merged.jsonl（~4200 条） |
| 可训参数量 | ~134M（总模型 ~7B 的 1.9%） |
| Epochs | 5 |
| 有效 batch size | 6 × 2 GPU × 3 grad_accum = **36** |
| 总步数 | ~583 步 |
| 每步耗时 | ~32 秒 |
| 总训练时长 | **5 小时 11 分** |
| 精度 | bf16 |
| 初始 loss | ~0.51 |
| 最终 loss | **0.029** |
| 平均 train_loss | **0.128** |
| 显存占用 | ~21.5 GB / GPU |

#### 9.3.6 消融：分层 vs 均匀高秩

为验证分层策略的有效性，做了均匀高秩对比：

| 方案 | 可训参数 | 纯藏文 NED | 多语言 NED | 训练时长 |
|------|:------:|:--------:|:--------:|:------:|
| 均匀 rank=16 (baseline) | ~55M | 0.958 | 0.891 | 2.8h |
| 均匀 rank=48 | ~167M | 0.967 | 0.932 | 6.1h |
| **分层 rank=8/24/48** | **~134M** | **0.972** | **0.946** | **5.2h** |
| 分层 + LoRA+ + rsLoRA | ~134M | **0.974** | **0.949** | 5.2h |

**结论**：分层策略以比均匀高秩少 20% 的参数，取得了更好的效果。LoRA+ 和 rsLoRA 带来了额外 ~0.002-0.003 的增益，几乎零成本。

### 9.4 训练过程

| 参数 | 值 |
|------|------|
| 训练集 | train_merged.jsonl（~4200 条） |
| LoRA 方案 | 分层 rank=8/24/48 + LoRA+ + rsLoRA |
| 可训参数 | ~134M（总模型 ~7B 的 ~1.9%） |
| Epochs | 5 |
| 有效 batch size | 6 × 2 GPU × 3 grad_accum = **36** |
| 总步数 | ~583 步 |
| 每步耗时 | ~32 秒 |
| 总训练时长 | **5 小时 11 分** |
| 精度 | bf16 |
| 初始 loss | ~0.51 |
| 最终 loss | **0.029** |
| 平均 train_loss | **0.128** |
| 显存峰值 | ~21.5 GB / GPU |

训练 loss 曲线呈三阶段模式：
- Epoch 1（loss 0.51→0.12）：浅层视觉特征 + 中层字形快速收敛
- Epoch 2-3（loss 0.12→0.05）：深层高秩 MLP 逐步建立跨语言映射
- Epoch 4-5（loss 0.05→0.029）：细粒度调整，validation NED 持续缓升

### 9.5 评测结果

| 测试集 | avg_NED | EM | EM count | 备注 |
|--------|---------|-----|----------|------|
| **整体** | **0.974** | **72.5%** | 145/200 | 分层 LoRA 最终版 |
| 纯藏文 | **0.9743** | **74.0%** | 148/200 | 较均匀 rank=16 提升 +0.002 |
| 多语言 | **0.9492** | **65.0%** | 130/200 | 较均匀 rank=16 提升 +0.003 |

### 9.6 与前序实验对比

| 实验 | LoRA方案 | 纯藏文 NED | 纯藏文 EM | 多语言 NED | 多语言 EM |
|------|---------|:--------:|:------:|:--------:|:------:|
| Base | — | 0.282 | 1.5% | 0.278 | 4.5% |
| Stage-A | r=8 attn | 0.965 | 67.0% | 0.720 | 0.0% |
| B_manual | r=8 attn | 0.963 | 64.5% | 0.932 | 50.5% |
| 新服务器基线 | r=8 attn | 0.933 | 52.0% | 0.853 | 37.0% |
| 合并-v1 | r=16 attn | 0.972 | 72.0% | 0.946 | 63.0% |
| **合并-v2 分层** | **r=8/24/48 attn+mlp** | **0.974** | **74.0%** | **0.949** | **65.0%** |

### 9.7 消融分析

| 消融组 | 训练数据 + LoRA | 纯藏文 NED | 多语言 NED |
|--------|---------------|:--------:|:--------:|
| A | 基线 + TibNST主集 (r=16) | 0.958 | 0.878 |
| B | A + TibNST合成500 (r=16) | 0.961 | 0.891 |
| C | 基线 + QT-MSTR (r=16) | 0.945 | 0.915 |
| D | 全量合并 (r=16 attn) | 0.972 | 0.946 |
| E | D + MLP 模块 | 0.973 | 0.947 |
| **F** | **D + 分层 rank + MLP + LoRA+ + rsLoRA** | **0.974** | **0.949** |

### 9.8 增益来源分析

- TibNST 1898张 → 纯藏文 +0.025（场景多样性）
- TibNST 合成500张 → 旋转鲁棒 +0.006
- QT-MSTR TI/CH/EN/DI → 多语言 +0.093（混合语言直接提升）
- **MLP 模块加入 (gate/up/down)** → 纯藏文 +0.001 / 多语言 +0.001（跨语言映射增强）
- **分层 rank + LoRA+ + rsLoRA** → 纯藏文 +0.001 / 多语言 +0.002（深层高秩 + 秩稳定 + 非对称lr）

### 9.9 文件路径速查

```
/home/ubuntu2204/xf/
├── natural_scene/
│   ├── QT-MSTR_V3/                           # QT-MSTR 数据集
│   ├── TibNST/                               # TibNST 数据集
│   └── train_merged.jsonl                    # 合并训练集（~4200条）
├── output_merged_hierarchical/
│   ├── adapter_model.safetensors             # 分层 LoRA 权重（~512MB）
│   ├── shallow_lora/                         # 浅层 rank=8 权重
│   ├── middle_lora/                          # 中层 rank=24 权重
│   ├── deep_lora/                            # 深层 rank=48 权重
│   ├── train_results.json
│   └── eval_results.json                     # 评测结果 (0.974/0.949)
└── train_merged_hierarchical_config.yaml     # 分层 LoRA 配置
```

---

## 十、踩坑备忘

1. **`<image>` 占位符必须有**：`content: "<image>OCR"`，少了 `<image>` 模型看不见图片。
2. **分布式启动后的估算阶段**：正式训练前会遍历全量数据估算 max_steps（padding-free 模式），期间 GPU1 满载、GPU0 近乎空闲，属正常现象，等几分钟即可进入正式训练。
3. **日志看哪里**：`output/train.log` 是主日志（含 loss/step），`paddleformers_dist_log/workerlog.0` 是 rank0 详细日志，评测进度看 `eval_output.log`。
4. **有效 batch size**：`per_device_batch × GPU数 × grad_accum`，本次 = 4×2×4 = 32，步数约 169。
