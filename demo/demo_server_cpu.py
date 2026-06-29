#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR-VL Tibetan/Multilingual OCR Demo Server
Port: 8899
Supports CPU-only environments by mocking use_triton_in_paddle
"""
import sys
import types

# Must mock use_triton_in_paddle BEFORE any paddle imports
_mock_cuda = types.ModuleType("use_triton_in_paddle.cuda")
def _no_cuda():
    raise RuntimeError("CUDA is not available on this server")
_mock_cuda.current_device = _no_cuda
_mock_triton = types.ModuleType("use_triton_in_paddle")
_mock_triton.cuda = _mock_cuda
sys.modules["use_triton_in_paddle"] = _mock_triton
sys.modules["use_triton_in_paddle.cuda"] = _mock_cuda

import os
import base64
import json
import time
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

MODEL_PATH = "/home/ubuntu2204/xf/output/export"
IMAGE_ROOT = "/home/ubuntu2204/xf/natural_scene/"
EVAL_PATH = "/home/ubuntu2204/xf/eval_results.json"

model = None
proc = None
gc = None

def load_model():
    global model, proc, gc
    print("[INFO] Loading model from " + MODEL_PATH, flush=True)
    import paddle
    paddle.set_device("cpu")
    print("[INFO] Using CPU device (bf16 auto-converted to fp32)", flush=True)
    from paddleformers.generation import GenerationConfig
    from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor

    model = AutoModelForConditionalGeneration.from_pretrained(MODEL_PATH, convert_from_hf=True, dtype="float32").eval()
    for m in [model.config, model.visual.config]:
        try:
            m._attn_implementation = "flashmask"
        except Exception:
            pass
    proc = AutoProcessor.from_pretrained(MODEL_PATH)
    gc = GenerationConfig(do_sample=False, bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=True)
    print("[INFO] Model ready!", flush=True)


def infer_pil(pil_img):
    import paddle
    msgs = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": "OCR"}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
    with paddle.no_grad():
        gid = model.generate(**inp, generation_config=gc, max_new_tokens=512)
        result = proc.decode(gid[0].tolist()[0], skip_special_tokens=True).strip()
    return result


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/ocr", methods=["POST"])
def ocr():
    t0 = time.time()
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field (base64)"}), 400
        img_bytes = base64.b64decode(data["image"])
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        if model is None:
            return jsonify({"error": "Model not ready"}), 503
        result = infer_pil(pil_img)
        elapsed = round(time.time() - t0, 2)
        return jsonify({"text": result, "time_sec": elapsed, "model": "PaddleOCR-VL-1.5 + LoRA"})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/demo_samples", methods=["GET"])
def demo_samples():
    try:
        with open(EVAL_PATH, encoding="utf-8") as f:
            d = json.load(f)
        results = d.get("results", [])
        top = sorted(results, key=lambda x: -x["similarity"])[:12]
        samples = []
        for r in top:
            img_path = os.path.join(IMAGE_ROOT, r["image"])
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                ext = r["image"].rsplit(".", 1)[-1].lower()
                mime = "jpeg" if ext in ("jpg", "jpeg") else ext
                samples.append({
                    "image": r["image"],
                    "image_b64": "data:image/" + mime + ";base64," + b64,
                    "ground_truth": r["ground_truth"],
                    "prediction": r["prediction"],
                    "similarity": round(r["similarity"], 4)
                })
        return jsonify({
            "avg_similarity": d["avg_similarity"],
            "exact_match_rate": d["exact_match_rate"],
            "total": d["total"],
            "samples": samples
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8899, debug=False)
