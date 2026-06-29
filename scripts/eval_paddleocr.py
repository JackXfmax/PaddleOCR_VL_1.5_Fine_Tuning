import argparse
import json
import os
import sys
from datetime import datetime

import paddle
from PIL import Image
from paddleformers.generation import GenerationConfig
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print("[%s] %s" % (ts, msg), flush=True)


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def ned(a, b):
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / max_len


def normalize(text):
    return text.strip().replace(' ', '')


MODEL_PATH = '/home/ubuntu2204/xf/output/export'
DATA_PATH = '/home/ubuntu2204/xf/natural_scene/test.jsonl'
IMG_DIR = '/home/ubuntu2204/xf/natural_scene'
OUTPUT_PATH = '/home/ubuntu2204/xf/eval_results.json'

log("Loading model from %s" % MODEL_PATH)
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH,
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

processor = AutoProcessor.from_pretrained(MODEL_PATH)
generation_config = GenerationConfig(
    do_sample=False,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
    use_cache=True,
)

results = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        obj = json.loads(line)
        image_name = obj['images'][0]
        image_path = os.path.join(IMG_DIR, image_name)
        gt = obj['messages'][1]['content'].strip()

        image = Image.open(image_path).convert('RGB')
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': 'OCR'},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pd',
        )
        with paddle.no_grad():
            generated_ids = model.generate(
                **inputs,
                generation_config=generation_config,
                max_new_tokens=512,
            )
            generated_ids = generated_ids[0].tolist()[0]
            pred = processor.decode(generated_ids, skip_special_tokens=True).strip()

        gt_norm = normalize(gt)
        pred_norm = normalize(pred)
        sim = ned(gt_norm, pred_norm)
        em = 1.0 if gt_norm == pred_norm else 0.0

        results.append({
            'image': image_name,
            'ground_truth': gt,
            'prediction': pred,
            'similarity': round(sim, 4),
            'exact_match': em,
        })

        if (idx + 1) % 20 == 0 or idx < 3:
            log("  [%d] sim=%.4f em=%.1f | %s" % (idx + 1, sim, em, image_name))

avg_sim = sum(r['similarity'] for r in results) / len(results)
avg_em = sum(r['exact_match'] for r in results) / len(results)

log("")
log("=== EVAL RESULTS ===")
log("Total: %d samples" % len(results))
log("Avg NED (similarity): %.4f" % avg_sim)
log("Exact Match: %.2f (%d/%d)" % (avg_em, int(sum(r['exact_match'] for r in results)), len(results)))

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump({
        'avg_similarity': round(avg_sim, 4),
        'exact_match_rate': round(avg_em, 4),
        'total': len(results),
        'results': results,
    }, f, ensure_ascii=False, indent=2)

log("Results saved to %s" % OUTPUT_PATH)
