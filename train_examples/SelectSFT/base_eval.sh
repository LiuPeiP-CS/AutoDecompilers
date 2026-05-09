#!/bin/bash

BIT=64
INDEX=0

export CUDA_VISIBLE_DEVICES=0,1,2,3
# 定义基础目录
DIRA="/XXX/models"  # 源目录,1.3B和6.7B下面的所有模型
DIRB="all_models_eval_results/"  # 目标目录
ALLMETRICS="all_models_eval_results/all_metrics.txt"

# 检查目录是否存在
[ ! -d "$DIRA" ] && { echo "错误: DIRA不存在: $DIRA"; exit 1; }
mkdir -p "$DIRB"

# 方式1：动态扫描所有子目录（自动发现）
echo "========== 方式1: 动态扫描所有子目录 =========="
shopt -s nullglob
#for folder in "$DIRA"/checkpoint-*/; do
for folder in "$DIRA"/*/; do
    if [ -d "$folder" ]; then
        MODEL_PATH="$folder"
        MODEL_NAME=$(basename "$folder")
        AGENT_OUTDIR="$DIRB/$MODEL_NAME"

        mkdir -p "$AGENT_OUTDIR"
        echo "============================创建: $AGENT_OUTDIR========================="
    fi

    ############################
    # 1. agent.py（只负责生成）
    ############################
    python agent.py \
      --workdir ${AGENT_OUTDIR} \
      --model_path ${MODEL_PATH} \
      -b ${BIT} \
      -i ${INDEX}

    if [ $? -ne 0 ]; then
      echo "[ERROR] agent.py failed for ${MODEL_NAME}"
      continue
    fi

    ############################
    # 2. recompile.py（自己保存评估结果）
    ############################
    python recompile.py \
      --workdir ${AGENT_OUTDIR} \
      --model ${MODEL_NAME} \
      -b ${BIT} \
      -i ${INDEX} \
      --all_metrics ${ALLMETRICS}

    if [ $? -ne 0 ]; then
      echo "[ERROR] recompile.py failed for ${MODEL_NAME}"
    fi

done
shopt -u nullglob
echo "All models finished."
