#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

echo "=== Tokenizer __init__ ==="
sed -n '40,70p' strhub/data/utils.py

echo ""
echo "=== CharsetAdapter ==="
grep -n 'CharsetAdapter\|whitespace\|strip\|replace' strhub/data/utils.py | head -20
