#!/bin/bash
sleep 120
echo "=== 检查checkpoint（2分钟后）==="
ls -la /home/xufei/parseq/outputs/parseq-tibetan/_/checkpoints/ 2>/dev/null || echo "No checkpoints yet"
echo ""
echo "=== GPU状态 ==="
nvidia-smi | grep -E 'MiB|Util'
echo ""
echo "=== 日志末尾 ==="
tail -10 /home/xufei/parseq/tibetan_train.log
