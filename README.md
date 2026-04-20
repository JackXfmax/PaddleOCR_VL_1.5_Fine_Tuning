# PaddleOCR-VL 藏文/多语言 LoRA 微调全流程文档
# This project was established for the Paddle Global Model Derivative Competition.


---

## 一、环境信息

| 项目 | 内容 |
|------|------|
| 系统 | Ubuntu 22.04，Python 3.10.12 |
| 硬件 | 2× RTX 4090
| Conda 环境：paddle
| 基座模型 | PaddleOCR-VL-1.5

---

## 二、数据准备

### 2.1 原始数据

根据上次收到的关于prompt的回复 我们这次处理数据，将各种数据集的标注prompt统一修改为：OCR

原始每条样本核心字段：
```json
{
  "image": "图片文件名",
  "merged_label_candidate": "最终转写答案（藏文+中文，含换行分段）",
  "has_auto_chinese": true
}
```

### 2.2 训练/测试集划分

按 9：1 划分，random_state=42：

```python
from sklearn.model_selection import train_test_split
import json

data = [json.loads(l) for l in open("reviewed_raw.jsonl")]
train, test = train_test_split(data, test_size=0.1, random_state=42)
# 训练集：1800 条，测试集：200 条
```

| 文件 | 条数 | 用途 |
|------|------|------|
| `train_joint_full.jsonl` | 1800 | 训练（本次使用） |
| `test_all.jsonl` | 200 | 评测（本次使用） |

### 2.3 数据格式转换

根据反馈PaddleOCR-VL-1.5 要求 user 消息里必须以 `<image>` 开头，prompt 使用 `OCR`：

```python
def to_paddle_format(item, image_root):
    return {
        "messages": [
            {
                "role": "user",
                "content": "<image>OCR"          
            },
            {
                "role": "assistant",
                "content": item["merged_label_candidate"]   
            }
        ],
        "images": [os.path.join(image_root, item["image"])]
    }
```
### 为了增强模型的鲁棒性，我们这次在微调训练的时候引入动态数据增强，以提升模型在复杂环境下的拟合效果
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

### 4.2 训练启动命令： 多卡训练

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
| 训练集 | train_joint_full.jsonl（1800 条，多语言数据） |
| 总Epochs | 3 |
| 有效 batch size | 4 × 2 GPU × 4 grad_accum = 32 |
| 总步数 | 169 步 |
| 每步耗时 | ~6 秒 |
| 总训练时长 | 16 分 16 秒 |
| 精度 | bf16 |
| 初始 loss | ~0.313 |
| 最终 loss | 0.056 |
| 平均 train_loss | 0.143 |

输出目录 `/home/ubuntu2204/xf/output/`：
```
output/
├── adapter_config.json           # LoRA 结构配置
├── adapter_model.safetensors     # LoRA 权重：约14.5 MB
├── tokenizer/
├── train_results.json
├── all_results.json
└── train.log
```

---

## 六、评测

### 6.1 评测指标

- Avg NED：归一化编辑距离相似度（SequenceMatcher ratio）
- EM：完全匹配率（Exact Match），pred 与 gt 字符串完全一致

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
| Avg NED | 0.9330 |
| Exact Match | 52.0%（104/200） |
| 测试集 | test_all.jsonl（200 条，多语言） |

### 值得说明的是：不知道是做数据增强的原因还是修改官方推荐的prompt 本次的训练结果 平均相似度比上次增加了0.012% 而EM提升了2个百分点

评测结果文件：`/home/ubuntu2204/xf/eval_results.json`（每条含 pred / gt / sim）

---

## 八、项目整体文件架构

```
/home/ubuntu2204/xf/
├── PaddleOCR-VL-1.5/                   # 基座模型
├── natural_scene/
│   ├── reviewed_raw.jsonl               # 原始 2000 条（备份）
│   ├── train_joint_full.jsonl           # 训练集（1800 条）
│   └── test_all.jsonl                   # 测试集（200 条）
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

## 九、补充说明

关于本次改进后我们只进行了多语言（藏文+汉语）的微调，效果确实比上次的要好，
我们的多语言数据集，在真实场景下可以很好的迁移到路牌等识别工作。这也是我们下一步打算要做的工作，我们很想补充真实路牌的数据，但是路牌一般都包含汉+藏+英+阿拉伯数字，所以我们的多语言数据集
突出了很大的贡献，因为路牌的收集难度虽然不大，但是跑遍整个拉萨也很难收集几百张数据，毕竟拉萨很小再加上我们的能力有限，所以迁移是一个很好的方法，也能达到很好的效果，所以我觉得，我们的数据
在微调后的paddle可以很好的应用到少数民族的导航和路牌识别等工作上，可以很好的帮助游客等外地人或服务本地人

2026/4/18

为了补充数据多样性和难度，我们这次还是补充了仅藏文的单语言数据集，数据集的构成：图片里只有藏文，标注也是纯藏文，
这个数据集在针对藏文单项训练的时候能达到很棒的效果，在真实场景下的识别率比多语言要高很多，针对这个数据集可以微调出来一个识别藏文的专项模型
2026/4/18

上次好像提供过二值化处理后的真实藏文手写古籍，但是那个是txt文件类型的标注文件，这次我们提供整合好的jsonl文件。
文件格式为：
```json
{"messages": [
    
    {"role": "user", 

"content": "<image>OCR"},

 {"role": "assistant",

  "content": "བརྒྱད་ཅུ་སྙེད་དང་། ཡང་ཐབས་གཅིག་ཏུ་གཤེགས་ཏེ་། བྱོན་ནས་ཚལ་ཆེན་པོར་། གཤེགས་པ་དང་།།"}], 
  
 "images": ["/home/xufei/acent/lines/I2KG2290310004_6.jpg"]
 }
```
2026/4/19补充
再次补充3W+张 经过二值化处理的真实场景下的藏文手写数字数据集，这个数据集只有图片，图片的名称就是对应的label
