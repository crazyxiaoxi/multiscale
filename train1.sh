#!/bin/bash

# 设置数据集名称和训练轮数
DATASET="origin"
EPOCHS=200
DEVICE="5"

# 模型版本列表
MODELS=("s" "spconv" )

# 创建日志目录
LOGDIR="logs"
mkdir -p $LOGDIR

# 依次训练每个模型版本
for MODEL in "${MODELS[@]}"; do
    echo "正在训练模型 yolov8${MODEL}，数据集：${DATASET}，轮数：${EPOCHS}"
    LOGFILE="${LOGDIR}/${DATASET}_ep${EPOCHS}_${MODEL}.log"
    python /home/xijiawen/code/ultralytics/train.py \
        --data $DATASET \
        --model $MODEL \
        --device $DEVICE \
        --ep $EPOCHS \
        2>&1 | tee "$LOGFILE"
done
