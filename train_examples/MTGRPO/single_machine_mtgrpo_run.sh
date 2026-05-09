#!/bin/bash

# ==========================================
# 1. 接收命令行传入的参数 (赋予默认值)
# ==========================================
NUM_PROCESSES=${num_processes:-2}
LLM_SIZE=${llm_size:-"1.3"}
LLM_STEP=${llm_step:-"14000"}
PROGRESS_ALPHA=${progress_alpha:-"0.15"}
BATCH_SIZE=${batch_size:-4}
DEC_MODE=${dec_mode:-"e2e"}        # ref仅仅针对6.7B模型
DATASET_PATH=${dataset_path:-"rl4096train4end2end.jsonl"}
NUM_ITERATIONS=${num_iterations:-1}   # epoch
TRAIN_ID=${train_id:-0}
MAX_TURNS=${max_turns:-2}

# 布尔开关变量 (默认开启)
ONLY_FINAL_TURN_REWARD=${only_final_turn_reward:-"true"}
SAME_ADV=${same_adv:-"true"}

# 动态拼接模型路径
SFT_MODEL_DIR="/workspace/qwen3-train/LLaMA-Factory-main/CIM-${LLM_SIZE}B-FullSFT-checkpoints/checkpoint-${LLM_STEP}"

# ==========================================
# 2. 处理 action="store_true" 的布尔参数
# ==========================================
OPTIONAL_ARGS=""

if [ "$ONLY_FINAL_TURN_REWARD" = "true" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --only_final_turn_reward"
    echo "🔍 状态: 已启用 only_final_turn_reward"
else
    echo "🔍 状态: 未启用 only_final_turn_reward"
fi

# 处理 same_adv
if [ "$SAME_ADV" = "true" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --same_adv"
    echo "🔍 状态: 已启用 same_adv"
else
    echo "🔍 状态: 未启用 same_adv"
fi

# ==========================================
# 3. 打印配置概览
# ==========================================
echo "=== 🚀 MTGRPO 训练配置参数 ==="
echo "进程数 (num_processes)       : $NUM_PROCESSES"
echo "模型参数量 (llm_size)            : $LLM_SIZE"
echo "模型步数 (llm_step)              : $LLM_STEP"
echo "SFT模型路径                      : $SFT_MODEL_DIR"
echo "Progress Alpha                   : $PROGRESS_ALPHA"
echo "单卡 Batch Size                  : $BATCH_SIZE"
echo "数据集模式 (dec_mode)              : $DEC_MODE"
echo "数据集路径 (dataset_path)        : $DATASET_PATH"
echo "训练轮数（epoch）                 :$NUM_ITERATIONS"
echo "==============================="

# ==========================================
# 4. 启动 Accelerate 训练
# ==========================================
accelerate launch \
    --config_file /workspace/trl-main/examples/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes "$NUM_PROCESSES" \
    MTGRPO_train.py \
    --abs_path /workspace/trl-main \
    --llm_size "$LLM_SIZE" \
    --llm_step "$LLM_STEP" \
    --sft_model_dir "$SFT_MODEL_DIR" \
    --judge_c \
    --judge_c_weight 0.5 \
    --re_exe_reward \
    --re_exe_reward_weight 1.0 \
    --syntax_reward \
    --syntax_reward_weight 0.3 \
    --semantic_reward \
    --semantic_reward_weight 0.2 \
    --max_turns "$MAX_TURNS" \
    --progress_alpha "$PROGRESS_ALPHA" \
    --num_generations 4 \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --grad_accum_steps 4 \
    --save_model_ratio 0.2 \
    --beta 0.02 \
    --vllm_tensor_parallel_size 1 \
    --dec_mode "$DEC_MODE" \
    --dataset_path "$DATASET_PATH" \
    --num_iterations "$NUM_ITERATIONS" \
    --train_id "$TRAIN_ID" \
    $OPTIONAL_ARGS
