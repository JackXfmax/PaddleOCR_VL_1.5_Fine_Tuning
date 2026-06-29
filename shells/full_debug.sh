#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

echo "=== CharsetAdapter ==="
sed -n '28,45p' strhub/data/utils.py

echo ""
echo "=== BaseSystem init ==="
sed -n '60,80p' strhub/models/base.py

echo ""
echo "=== 完整HYDRA_FULL_ERROR测试 ==="
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 timeout 30 python train.py \
  --config-name=main \
  data.img_size=[32,512] \
  model.img_size=[32,512] \
  model.max_label_length=200 \
  data.max_label_length=200 \
  trainer.max_epochs=1 \
  trainer.devices=1 \
  data.batch_size=4 \
  model.batch_size=4 2>&1 | tail -40
