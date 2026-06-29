import base64
# The eval script content (pure Python, no shell escaping issues)
script = r"""import json, os
from datetime import datetime
import paddle
from PIL import Image
from paddleformers.generation import GenerationConfig
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor

def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)

def levenshtein(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j-1]+1, prev[j]+1, prev[j-1]+(0 if ca==cb else 1)))
        prev = cur
    return prev[-1]

def ned(a, b):
    ml = max(len(a), len(b))
    return 1.0 if ml == 0 else 1.0 - levenshtein(a, b) / ml

def norm(t):
    return t.strip().replace(" ", "")

MP = "/home/ubuntu2204/xf/output/export"
DP = "/home/ubuntu2204/xf/natural_scene/test.jsonl"
ID = "/home/ubuntu2204/xf/natural_scene"
OP = "/home/ubuntu2204/xf/eval_results.json"

log("Loading model from " + MP)
model = AutoModelForConditionalGeneration.from_pretrained(MP, convert_from_hf=True).eval()
for m in [model.config, model.visual.config]:
    try:
        m._attn_implementation = "flashmask"
    except:
        pass

proc = AutoProcessor.from_pretrained(MP)
gc = GenerationConfig(do_sample=False, bos_token_id=1, eos_token_id=2, pad_token_id=0, use_cache=True)

results = []
with open(DP, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        obj = json.loads(line)
        iname = obj["images"][0]
        ipath = os.path.join(ID, iname)
        gt = obj["messages"][1]["content"].strip()
        img = Image.open(ipath).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "OCR"}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
        with paddle.no_grad():
            gid = model.generate(**inp, generation_config=gc, max_new_tokens=512)
            pred = proc.decode(gid[0].tolist()[0], skip_special_tokens=True).strip()
        gn, pn = norm(gt), norm(pred)
        s = ned(gn, pn)
        e = 1.0 if gn == pn else 0.0
        results.append({"image": iname, "ground_truth": gt, "prediction": pred, "similarity": round(s, 4), "exact_match": e})
        if (idx + 1) % 20 == 0 or idx < 3:
            log("  [%d] sim=%.4f em=%.1f | %s" % (idx + 1, s, e, iname))

avs = sum(r["similarity"] for r in results) / len(results)
ave = sum(r["exact_match"] for r in results) / len(results)
log("")
log("=== EVAL RESULTS ===")
log("Total: %d samples" % len(results))
log("Avg NED: %.4f" % avs)
log("Exact Match: %.2f (%d/%d)" % (ave, int(sum(r["exact_match"] for r in results)), len(results)))
with open(OP, "w", encoding="utf-8") as f:
    json.dump({"avg_similarity": round(avs, 4), "exact_match_rate": round(ave, 4), "total": len(results), "results": results}, f, ensure_ascii=False, indent=2)
log("Done: " + OP)
"""

encoded = base64.b64encode(script.encode()).decode()
print(encoded)
