#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
from strhub.models.base import _load_charset
from strhub.data.utils import Tokenizer

charset_path = '/home/xufei/tibet_acent/tibetan_charset_final.txt'
charset_str = _load_charset(charset_path)
print(f'charset len: {len(charset_str)}')
print(f'Has space: {" " in charset_str}')

# 测试Tokenizer
tok = Tokenizer(charset_str)
print(f'Tokenizer vocab size: {len(tok._stoi)}')
print(f'Space in stoi: {" " in tok._stoi}')

# 测试encode
test = 'ཀ ཁ'
encoded = tok.encode([test], 'cpu')
print(f'Encoded: {encoded}')
PYEOF
