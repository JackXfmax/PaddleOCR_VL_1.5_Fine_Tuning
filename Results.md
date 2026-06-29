# 🏔️ TibetanOCR — PaddleOCR-VL Tibetan Scene Text Recognition

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.0.0-0066CC)](https://www.paddlepaddle.org.cn/)
[![LoRA](https://img.shields.io/badge/LoRA-Hierarchical-orange)](#)
[![SOTA](https://img.shields.io/badge/Tibetan%20NED-0.974-brightgreen)](#)

**面向藏文自然场景文本的端到端 OCR 微调方案**  
PaddleOCR-VL-1.5B + 分层 LoRA + 多语言混合训练

</div>

---

## 📊 Performance

| Test Set | avg_NED | EM | LoRA Scheme |
|----------|:------:|:--:|------------|
| **Tibetan (Pure)** | **0.974** | **74%** | Hierarchical r=8/24/48 |
| **Multilingual** | **0.949** | **65%** | attn + MLP + LoRA+ + rsLoRA |

| Ablation | Tibetan NED | Multilingual NED | Params | Time |
|----------|:----------:|:----------------:|:------:|:----:|
| Base (PaddleOCR-VL-1.5B) | 0.282 | 0.278 | — | — |
| Stage-A (1800 Tibetan) | 0.965 | 0.720 | ~28M | 1.3h |
| B_manual (3600 Mixed) | 0.963 | 0.932 | ~28M | 2.6h |
| Merged r=16 attn-only | 0.972 | 0.946 | ~55M | 3.5h |
| **Merged Hierarchical v2** | **0.974** | **0.949** | ~134M | 5.2h |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│        TibetanOCR-Hierarchical-LoRA v2       │
├─────────────────────────────────────┤
│  Shallow (L0-10):  r=8  attn only           │  ← preserve visual features
│  Middle  (L11-21): r=24 attn + MLP           │  ← cross-lingual glyph mapping
│  Deep    (L22-31): r=48 attn + MLP           │  ← OCR task decoding
├─────────────────────────────────────┤
│  Enhancements:                                │
│  • LoRA+ (asymmetric lr: λ=1.5/2.0/2.5)      │
│  • rsLoRA (rank-stabilized: α/√r)            │
│  • Cosine scheduler + label smoothing        │
└─────────────────────────────────────┘
```

---

## 📁 Project Structure

```
TibetanOCR/
├── README.md                          # ← you are here
├── DEVELOPER_GUIDE.md                 # Full developer documentation (1820 lines)
├── CODE_OF_CONDUCT.md                 # Contributor Covenant 2.1
├── CONTRIBUTING.md                    # Contribution guide (Conventional Commits)
├── .env.example                       # Environment template
├── requirements.txt                   # Python dependencies
├── train_config.yaml                  # LoRA training config
├── Dockerfile                         # Docker deployment
│
├── docs/                              # 📚 Documentation
│   ├── TibetanOCR_Hierarchical_LoRA_v2_方案设计书.md
│   ├── PaddleOCR_VL_新服务器微调全流程_2026-04-17.md
│   ├── 实验结果汇总_2026-04-07.md
│   ├── 研究报告_智能藏文路牌识别与导航系统.md
│   ├── crnn_training_summary_2026-04-10.md
│   └── 第4章_实验设计与结果分析.md
│
├── scripts/                           # 🐍 Core training & evaluation
│   ├── paddleocr_vl_v15_template.py   # Training data template
│   ├── full_eval.py                   # Full evaluation pipeline
│   ├── infer_ocr.py                   # Single image inference
│   ├── eval_paddleocr.py              # Evaluation entry point
│   ├── eval_paddleocr_v2.py           # Evaluation v2
│   ├── merge_and_eval.py              # Dataset merge + evaluation
│   ├── gen_eval_b64.py                # Base64 encoding for server eval
│   ├── process_reviewed.py            # Review data processing
│   ├── make_charset.py                # Character set generation
│   ├── make_script.py                 # Training script generator
│   ├── build_multilingual_experiment_sets.py
│   ├── convert_acent_to_jsonl.py      # Annotation format converter
│   ├── convert_dataset_multilingual.py # Multilingual dataset converter
│   ├── supplement_chinese_labels.py   # Chinese label supplement
│   ├── check_eval.py / check_gpu.py / check_ckpt.py  # Diagnostics
│
├── crnn/                              # 🔬 CRNN experiments
│   ├── crnn_train.py / crnn_train_v2.py
│   ├── crnn_v6_final.py / crnn_v5_improved.py
│   ├── crnn_v4_upload.py
│   ├── crnn_fixed.py / base_fixed.py / module_fixed.py
│
├── demo/                              # 🎨 Demo & Web UI
│   ├── demo.html                      # Static offline demo (no server needed)
│   ├── demo_showcase_final.html       # Full showcase page
│   ├── roadsign.html                  # Road sign recognition demo
│   ├── demo_server.py                 # Flask server (GPU - PaddlePaddle)
│   ├── demo_server_cpu.py             # CPU inference server
│   ├── app.py                         # API server
│   ├── serve_roadsign.js              # Node.js road sign server
│   └── serve_demo.ps1                 # PowerShell launcher
│
├── miniapp/                           # 📱 WeChat Mini Program
│   └── roadsign_miniapp/
│       ├── app.js / app.json / app.wxss
│       ├── pages/index/
│       ├── utils/amap.js / api.js / dict.js
│       └── project.config.json
│
├── results/                           # 📊 Experiment results
│   ├── eval_results.json              # New server baseline (v1)
│   ├── eval_merged_results.json       # Merged training results (v2)
│   ├── train_results.json
│   └── all_results.json
│
└── shells/                            # 🐚 Server-side training scripts
    ├── master_pipeline_BC.sh          # End-to-end training pipeline
    ├── start_train.sh
    └── ... (25 shell scripts)
```

---

## 🚀 Quick Start

### 1. Environment

```bash
# GPU server (recommended)
conda create -n tibetanocr python=3.10 -y
conda activate tibetanocr
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Convert annotations to JSONL format
python scripts/convert_acent_to_jsonl.py --input data/raw/ --output data/train.jsonl

# Build multilingual experiment sets
python scripts/build_multilingual_experiment_sets.py
```

### 3. Training

```bash
# Copy config and start training
cp train_config.yaml ./output/
bash shells/start_train.sh
```

### 4. Evaluation

```bash
python scripts/full_eval.py \
    --model_path ./output/adapter_model.safetensors \
    --test_set data/test.jsonl \
    --output results/eval_results.json
```

### 5. Demo (No Server Needed!)

Just open `demo/demo.html` in your browser.

Or start the Flask server:

```bash
python demo/demo_server.py  # GPU version (PaddlePaddle required)
python demo/demo_server_cpu.py  # CPU version
```

---

## 📚 Datasets

| Dataset | Scene | Images | Annotations | Languages |
|---------|-------|:------:|:-----------:|-----------|
| QT-MSTR V3 | Storefront | 1,000 | 12,336 boxes | TI/CH/EN/DI |
| TibNST | Natural scene | 3,793 | 2,046 entries | Tibetan |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Citation

If you find this work useful, please cite:

```bibtex
@misc{TibetanOCR2026,
  title     = {TibetanOCR: PaddleOCR-VL Hierarchical LoRA Fine-tuning for Tibetan Scene Text Recognition},
  year      = {2026},
  url       = {https://github.com/YOUR_USERNAME/TibetanOCR}
}
```

---

<div align="center">
  Made with ❤️ for preserving Tibetan language in the digital age
</div>
