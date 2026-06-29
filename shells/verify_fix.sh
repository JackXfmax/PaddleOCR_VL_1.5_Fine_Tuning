#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

echo "=== 检查base.py修改 ==="
grep -n '_load_charset\|Tokenizer\|CTCToken' strhub/models/base.py | grep -v '#'

echo ""
echo "=== 清除pyc缓存 ==="
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "Cache cleared"
