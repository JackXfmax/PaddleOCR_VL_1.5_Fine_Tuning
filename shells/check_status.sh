#!/bin/bash
echo "=== 进程状态 ==="
ps aux | grep -E 'train\.py|python' | grep -v grep | grep -v defunct

echo ""
echo "=== GPU状态 ==="
nvidia-smi | grep -E 'MiB|%'

echo ""
echo "=== 日志末尾 ==="
tail -5 /home/xufei/parseq/tibetan_train.log
