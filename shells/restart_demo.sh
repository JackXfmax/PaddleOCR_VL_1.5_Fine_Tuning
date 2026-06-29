#!/bin/bash
# Restart demo_server with CPU float32 support
# Kill old server
kill $(ps aux | grep demo_server | grep python | grep -v grep | awk '{print $2}') 2>/dev/null
sleep 1
# Start new server
source /home/ubuntu2204/miniconda3/etc/profile.d/conda.sh
conda activate paddle
nohup python3 /home/ubuntu2204/xf/demo_server.py > /home/ubuntu2204/xf/demo_server.log 2>&1 &
echo "PID=$!"
