#!/bin/bash
source /home/ubuntu2204/miniconda3/etc/profile.d/conda.sh
conda activate paddle
python3 /tmp/test_cpu_v2.py > /tmp/cpu_test_out.txt 2>&1 &
echo "PID=$!"
