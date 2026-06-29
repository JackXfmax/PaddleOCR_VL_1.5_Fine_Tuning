#!/usr/bin/env python3
import json
f = "/home/xufei/tibetan_ocr_lora/stage2_manual_20260406/eval_B_multilingual_manual_20260406.json"
d = json.load(open(f))
print("Type:", type(d))
if isinstance(d, dict):
    print("Keys:", list(d.keys()))
    if "results" in d:
        print("Results count:", len(d["results"]))
        r = d["results"][0]
        print("First result keys:", list(r.keys()))
        print("Image:", r.get("image", "N/A"))
elif isinstance(d, list):
    print("Length:", len(d))
    if d:
        print("First item keys:", list(d[0].keys()))
