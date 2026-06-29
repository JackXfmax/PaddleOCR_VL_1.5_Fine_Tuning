#!/bin/bash
source /home/xufei/miniconda3/etc/profile.d/conda.sh
conda activate acent
cd /home/xufei/parseq

# 清理旧输出
rm -rf outputs/parseq-tibetan/_

# 后台启动训练
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --config-name=main \
  data.img_size=[32,512] \
  model.img_size=[32,512] \
  model.max_label_length=200 \
  data.max_label_length=200 \
  trainer.max_epochs=50 \
  trainer.devices=1 \
  data.batch_size=32 \
  model.batch_size=32 \
  > /home/xufei/parseq/tibetan_train.log 2>&1 &

echo "Training PID: $!"
