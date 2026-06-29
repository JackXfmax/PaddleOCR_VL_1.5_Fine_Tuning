#!/bin/bash
python3 << 'PYEOF'
with open('/home/xufei/parseq/configs/main.yaml', 'r') as f:
    c = f.read()
c = c.replace('train_dir: .', 'train_dir: train')
with open('/home/xufei/parseq/configs/main.yaml', 'w') as f:
    f.write(c)
print('fixed train_dir')
PYEOF
grep 'train_dir' /home/xufei/parseq/configs/main.yaml
