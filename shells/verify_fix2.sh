#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

# 验证
echo "=== 验证tibetan.yaml ==="
cat configs/charset/tibetan.yaml

# 清pyc
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "Cache cleared"

# 快速测试
echo "=== 快速导入测试 ==="
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

with initialize_config_dir(config_dir='/home/xufei/parseq/configs', version_base='1.2'):
    cfg = compose(config_name='main', overrides=['data.batch_size=4','model.batch_size=4'])
    from hydra.utils import instantiate
    model = instantiate(cfg.model)
    print(f"Tokenizer size: {len(model.tokenizer._stoi)}")
    print(f"Space in tokenizer: {' ' in model.tokenizer._stoi}")
PYEOF
