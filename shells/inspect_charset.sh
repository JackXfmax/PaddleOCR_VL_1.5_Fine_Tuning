#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

echo "=== _load_charset 实现 ==="
sed -n '18,30p' strhub/models/base.py

echo ""
echo "=== 用xxd查看字符集文件内容（前20行） ==="
head -5 /home/xufei/tibet_acent/tibetan_charset_final.txt | xxd | head -30

echo ""
echo "=== Python实际读取测试（打印每个字符repr） ==="
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

with open('/home/xufei/tibet_acent/tibetan_charset_final.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f'Total lines: {len(lines)}')
for i, line in enumerate(lines[:5]):
    print(f'Line {i}: {repr(line)}')
    
# 用_load_charset加载
from strhub.models.base import _load_charset
charset = _load_charset('/home/xufei/tibet_acent/tibetan_charset_final.txt')
print(f'\nLoaded charset len={len(charset)}')
print(f'First 5 chars: {[repr(c) for c in charset[:5]]}')
print(f'Space in charset: {" " in charset}')
PYEOF
