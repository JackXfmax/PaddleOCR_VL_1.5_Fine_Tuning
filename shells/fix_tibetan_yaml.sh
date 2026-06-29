#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

python3 << 'PYEOF'
# 读取字符集文件（包含所有字符含空格）
with open('/home/xufei/tibet_acent/tibetan_charset_final.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

chars = [line.rstrip('\n') for line in lines]
charset_str = ''.join(chars)

print(f'Charset len: {len(charset_str)}')
print(f'Has space: {" " in charset_str}')
print(f'Has tab: {chr(9) in charset_str}')

# 用YAML单引号字符串格式写入tibetan.yaml
# 注意：单引号字符串中要转义单引号为''
escaped = charset_str.replace("'", "''")

yaml_content = f"""# @package _global_
model:
  charset_train: '{escaped}'
"""

with open('configs/charset/tibetan.yaml', 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print('Updated tibetan.yaml')

# 验证
with open('configs/charset/tibetan.yaml', 'r', encoding='utf-8') as f:
    content = f.read()
print(f'File size: {len(content)} bytes')
PYEOF
