#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

# 模拟训练时的加载
from strhub.models.base import _load_charset

# main.yaml里的路径
path = '/home/xufei/tibet_acent/tibetan_charset_final.txt'
charset = _load_charset(path)
print(f'Loaded charset len: {len(charset)}')
print(f'Has space: {repr(" ")} in charset: {" " in charset}')
print(f'Chars: {[repr(c) for c in charset[:10]]}')

# 创建Tokenizer
from strhub.data.utils import Tokenizer
tok = Tokenizer(charset)
print(f'Tokenizer stoi keys (first 10): {list(tok._stoi.keys())[:10]}')
print(f'Space " " in stoi: {" " in tok._stoi}')

# 测试encode带空格的字符串
test_str = 'ཀ ཁ'
try:
    result = tok.encode([test_str], 'cpu')
    print(f'Encode success: {result}')
except Exception as e:
    print(f'Encode failed: {e}')
PYEOF
