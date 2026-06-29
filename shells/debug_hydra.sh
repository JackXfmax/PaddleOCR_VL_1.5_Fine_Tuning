#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
import os

# 模拟Hydra配置实例化
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir

with initialize_config_dir(config_dir='/home/xufei/parseq/configs', version_base='1.2'):
    cfg = compose(config_name='main', overrides=[
        'data.img_size=[32,512]',
        'model.img_size=[32,512]',
        'model.max_label_length=200',
        'data.max_label_length=200',
        'trainer.max_epochs=1',
        'data.batch_size=4',
        'model.batch_size=4',
    ])
    
    print("model.charset_train:", cfg.model.charset_train)
    print("data.charset_train:", cfg.data.charset_train)
    
    # 实例化模型
    from hydra.utils import instantiate
    model = instantiate(cfg.model)
    
    # 检查tokenizer
    print(f"\nTokenizer stoi size: {len(model.tokenizer._stoi)}")
    print(f"Space in tokenizer: {' ' in model.tokenizer._stoi}")
    print(f"First 5 keys: {list(model.tokenizer._stoi.keys())[:5]}")
PYEOF
