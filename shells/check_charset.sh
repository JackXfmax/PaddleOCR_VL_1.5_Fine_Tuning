#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
from strhub.models.base import _load_charset

path = '/home/xufei/tibet_acent/tibetan_charset_final.txt'
charset = _load_charset(path)
print(f'Total chars: {len(charset)}')
print(f'Has space: {" " in charset}')
print(f'Has tab: {chr(9) in charset}')

# 检查val集中的字符
import lmdb
env = lmdb.open('/home/xufei/tibet_acent/parseq_lmdb/val', readonly=True)
with env.begin() as txn:
    n = int(txn.get(b'num-samples').decode())
    print(f'Val samples: {n}')
    missing = set()
    for i in range(1, min(n+1, 50)):
        label = txn.get(f'label-{i:09d}'.encode())
        if label:
            label = label.decode('utf-8')
            for c in label:
                if c not in charset:
                    missing.add(repr(c))
    print(f'Missing chars in first 50: {missing}')
env.close()
PYEOF
