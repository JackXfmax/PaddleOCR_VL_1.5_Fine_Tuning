#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

python3 << 'PYEOF'
with open('strhub/models/base.py', 'r') as f:
    content = f.read()

# 修复 CrossEntropySystem.__init__
old = "        tokenizer = Tokenizer(charset_train)\n        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)"
new = "        tokenizer = Tokenizer(_load_charset(charset_train))\n        super().__init__(tokenizer, _load_charset(charset_test), batch_size, lr, warmup_pct, weight_decay)"

if old in content:
    content = content.replace(old, new)
    print("Fixed CrossEntropySystem")
else:
    print("WARNING: CrossEntropySystem pattern not found")

# 修复 CTCSystem.__init__
old2 = "        tokenizer = CTCTokenizer(charset_train)\n        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)"
new2 = "        tokenizer = CTCTokenizer(_load_charset(charset_train))\n        super().__init__(tokenizer, _load_charset(charset_test), batch_size, lr, warmup_pct, weight_decay)"

if old2 in content:
    content = content.replace(old2, new2)
    print("Fixed CTCSystem")
else:
    print("WARNING: CTCSystem pattern not found")

with open('strhub/models/base.py', 'w') as f:
    f.write(content)

print("Done")
PYEOF
