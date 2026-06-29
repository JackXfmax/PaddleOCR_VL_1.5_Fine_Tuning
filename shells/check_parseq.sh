#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

echo "=== PARSeq system.py init ==="
sed -n '30,75p' strhub/models/parseq/system.py

echo ""
echo "=== 确认base.py 198-200行 ==="
sed -n '193,205p' strhub/models/base.py
