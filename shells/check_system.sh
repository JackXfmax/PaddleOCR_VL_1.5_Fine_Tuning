#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

# 查看模型system.py中如何处理charset
grep -n 'charset' strhub/models/parseq/system.py | head -20
echo '---'
grep -n 'charset\|_load_charset' strhub/models/base.py | head -20
