#!/bin/bash

BIT=64
INDEX=0

#export CUDA_VISIBLE_DEVICES=0,1,2,3
BASE_PATH="/workspace/trl-main/train_examples/SelectMT"
DIRA="/workspace/trl-main/trained_checkpoints/GRPO/6.7-e2e"
DIRB="/workspace/trl-main/train_examples/SelectMT/all_models_eval_results/eb-6.7-e2e"
DATA_PATH="/workspace/trl-main/datasets/test/TestEBEnd2end.jsonl"
#ALLMETRICS="all_models_eval_results/all_metrics.txt"

[ ! -d "$DIRA" ] && { echo "错误: DIRA不存在: $DIRA"; exit 1; }
mkdir -p "$DIRB"



# 清理函数：杀掉所有残留的 gcc 和编译产物进程
cleanup() {
    echo "[CLEANUP] Killing residual gcc/compile processes..."
    # 杀掉所有未结束的 gcc 进程
    pkill -KILL -f "gcc" 2>/dev/null
    # 杀掉 tmp 目录下的可执行文件进程（recompile.py 产生的）
    pkill -KILL -f "combine" 2>/dev/null
    pkill -KILL -f "onlyfunc" 2>/dev/null
}

# 脚本退出时（正常/异常/Ctrl+C）都触发清理
trap cleanup EXIT INT TERM




echo "========== 扫描所有子目录 =========="
shopt -s nullglob

for folder in "$DIRA"/bestcheckpoint-*/; do        # 用 */ 匹配所有子目录，不限于 checkpoint-*
    [ -d "$folder" ] || continue       # 不是目录就跳过

    MODEL_PATH="$folder"
    MODEL_NAME=$(basename "$folder")
    AGENT_OUTDIR="$DIRB/$MODEL_NAME"

    mkdir -p "$AGENT_OUTDIR"
    echo "===== 处理模型: $MODEL_NAME ====="

    python "$BASE_PATH/MTGRPO_eval.py" \
      --model_dir              "$MODEL_PATH" \
      --data_path              "$DATA_PATH"  \
      --output_dir             "$AGENT_OUTDIR" \
      --tensor_parallel_size   8 \
      --dataset                exebench \
      --batch_size             8  \
      --max_turns              2  \


    if [ $? -ne 0 ]; then
        echo "[ERROR] MTGRPO_eval.py failed for ${MODEL_NAME}"
        continue
    fi

    echo "[DONE] $MODEL_NAME"


    RESULT_FILE=$(ls "$AGENT_OUTDIR"/*.jsonl 2>/dev/null | head -n 1)

    if [ -z "$RESULT_FILE" ]; then
        echo "[ERROR] 在 $AGENT_OUTDIR 中未找到结果文件，跳过 recompile"
        continue
    fi

    echo "[INFO] 使用结果文件: $RESULT_FILE"

    setsid python "$BASE_PATH/recompile_rl.py" \
      -i ${RESULT_FILE}


    RECOMPILE_PID=$!
    # 等待结束，超时可按需调整（单位：秒）
    wait $RECOMPILE_PID
    RECOMPILE_EXIT=$?

    # 每个模型跑完后，立刻清理该轮残留进程
    cleanup

    if [ $RECOMPILE_EXIT -ne 0 ]; then
      echo "[ERROR] recompile.py failed for ${MODEL_NAME}"
    fi
#    ############################
#    # 2. recompile.py（自己保存评估结果）
#    ############################
#    python recompile.py \
#      --workdir ${AGENT_OUTDIR} \
#      --model ${MODEL_NAME} \
#      -b ${BIT} \
#      -i ${INDEX} \
#      --all_metrics ${ALLMETRICS}
#
#    if [ $? -ne 0 ]; then
#      echo "[ERROR] recompile.py failed for ${MODEL_NAME}"
#    fi

done

shopt -u nullglob
echo "All models finished."