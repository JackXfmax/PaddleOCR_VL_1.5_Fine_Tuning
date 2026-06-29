#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

# 查看base.py第190-205行
sed -n '190,225p' strhub/models/base.py
