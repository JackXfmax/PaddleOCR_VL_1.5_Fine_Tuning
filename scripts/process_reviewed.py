import json
import os
import random

DATA_DIR = "/home/ubuntu2204/xf/natural_scene"
RAW_FILE = os.path.join(DATA_DIR, "reviewed_raw.jsonl")
random.seed(42)

# 1. Read raw data
raw_data = []
with open(RAW_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            raw_data.append(json.loads(line))

print("Total raw samples: %d" % len(raw_data))

# 2. Split into tibetan-only and multilingual (has_auto_chinese=True means multilingual)
tibetan_only = []
multilingual = []
for item in raw_data:
    has_chinese = item.get("has_auto_chinese", False)
    merged = item.get("merged_label_candidate", "")
    if has_chinese and merged:
        multilingual.append(item)
    else:
        tibetan_only.append(item)

print("Tibetan-only: %d" % len(tibetan_only))
print("Multilingual (with Chinese): %d" % len(multilingual))

# 3. Split train/test (90/10) for each category
random.shuffle(tibetan_only)
random.shuffle(multilingual)

tib_train = tibetan_only[:int(len(tibetan_only)*0.9)]
tib_test = tibetan_only[int(len(tibetan_only)*0.9):]
mul_train = multilingual[:int(len(multilingual)*0.9)]
mul_test = multilingual[int(len(multilingual)*0.9):]

print("Tibetan train: %d, test: %d" % (len(tib_train), len(tib_test)))
print("Multilingual train: %d, test: %d" % (len(mul_train), len(mul_test)))

# 4. Build official format: keep messages/images, change prompt to OCR
def to_official(item):
    obj = {
        "messages": [
            {"role": "user", "content": "<image>OCR"},
            {"role": "assistant", "content": item["merged_label_candidate"]}
        ],
        "images": item["images"]
    }
    return obj

# 5. Generate datasets
# Dataset A: Tibetan only train
with open(os.path.join(DATA_DIR, "train_tibetan.jsonl"), "w", encoding="utf-8") as f:
    for item in tib_train:
        f.write(json.dumps(to_official(item), ensure_ascii=False) + "\n")

# Dataset B: Joint full (tibetan + multilingual) train
joint_train = tib_train + mul_train
random.shuffle(joint_train)
with open(os.path.join(DATA_DIR, "train_joint_full.jsonl"), "w", encoding="utf-8") as f:
    for item in joint_train:
        f.write(json.dumps(to_official(item), ensure_ascii=False) + "\n")

# Dataset C: Multilingual only train
with open(os.path.join(DATA_DIR, "train_multilingual.jsonl"), "w", encoding="utf-8") as f:
    for item in mul_train:
        f.write(json.dumps(to_official(item), ensure_ascii=False) + "\n")

# Test sets
# Pure tibetan test
with open(os.path.join(DATA_DIR, "test_tibetan.jsonl"), "w", encoding="utf-8") as f:
    for item in tib_test:
        f.write(json.dumps(to_official(item), ensure_ascii=False) + "\n")

# Multilingual test (full evaluation set)
with open(os.path.join(DATA_DIR, "test_multilingual.jsonl"), "w", encoding="utf-8") as f:
    for item in mul_test:
        f.write(json.dumps(to_official(item), ensure_ascii=False) + "\n")

# Combined test (all)
combined_test = tib_test + mul_test
random.shuffle(combined_test)
with open(os.path.join(DATA_DIR, "test_all.jsonl"), "w", encoding="utf-8") as f:
    for item in combined_test:
        f.write(json.dumps(to_official(item), ensure_ascii=False) + "\n")

# Summary
print("\nGenerated files:")
for fname in ["train_tibetan.jsonl", "train_joint_full.jsonl", "train_multilingual.jsonl",
              "test_tibetan.jsonl", "test_multilingual.jsonl", "test_all.jsonl"]:
    fp = os.path.join(DATA_DIR, fname)
    cnt = sum(1 for _ in open(fp, encoding="utf-8"))
    print("  %s: %d lines" % (fname, cnt))

# Verify format
with open(os.path.join(DATA_DIR, "train_joint_full.jsonl"), "r", encoding="utf-8") as f:
    first = json.loads(f.readline().strip())
    print("\nSample format: %s" % json.dumps(first, ensure_ascii=False, indent=2))

# Clean up bak files
for bak in ["swift_data.jsonl.bak", "test.jsonl.bak", "train.jsonl.bak"]:
    bp = os.path.join(DATA_DIR, bak)
    if os.path.exists(bp):
        os.remove(bp)
        print("Removed: %s" % bak)
