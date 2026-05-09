#!/bin/bash

# ==========================================
# 1. 接收命令行传入的参数 (赋予默认值)
# ==========================================
NUM_PROCESSES=${num_processes:-2}
LLM_SIZE=${llm_size:-"1.3"}
LLM_STEP=${llm_step:-"14000"}
JUDGE_C_WEIGHT=${judge_c_weight:-0.5}
RE_EXE_REWARD_WEIGHT=${re_exe_reward_weight:-1.0}
SYNTAX_REWARD_WEIGHT=${syntax_reward_weight:-0.3}
SEMANTIC_REWARD_WEIGHT=${semantic_reward_weight:-0.2}
BATCH_SIZE=${batch_size:-16}
TRAIN_ID=${train_id:-0}

USE_APTED=${use_apted:-"false"}

# 动态拼接模型路径 (解决之前的语法错误)
SFT_MODEL_DIR="/workspace/qwen3-train/LLaMA-Factory-main/CIM-${LLM_SIZE}B-FullSFT-refcheckpoints/checkpoint-${LLM_STEP}"


OPTIONAL_ARGS=""

# 判断：如果传入的 use_apted 是 "true"，则把 --use_apted 拼接到参数里
if [ "$USE_APTED" = "true" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --use_apted"
    echo "🔍 检测到 use_apted=true，已启用 APTED 评估"
else
    echo "🔍 未启用 APTED 评估"
fi


# ==========================================
# 2. 设置网络与分布式环境变量
# ==========================================
export NCCL_P2P_DISABLE=1  # 如果是不同型号卡或网格架构，有时需要禁用 P2P
export NCCL_IB_DISABLE=1   # 如果没有 InfiniBand，强制禁用
export TOKENIZERS_PARALLELISM=false

echo "=== 🚀 Training Settings ==="
echo "num_processes       : $NUM_PROCESSES"
echo "llm_size        : $LLM_SIZE"
echo "llm_step          : $LLM_STEP"
echo "SFT model path                  : $SFT_MODEL_DIR"
echo "Judge C reward_weights                 : $JUDGE_C_WEIGHT"
echo "Re-exe reward_weights                  : $RE_EXE_REWARD_WEIGHT"
echo "Syntax reward_weights                  : $SYNTAX_REWARD_WEIGHT"
echo "Semantic reward_weights                : $SEMANTIC_REWARD_WEIGHT"
echo "Is APTED used                  : $USE_APTED"
echo "==========================="

# ==========================================
# 3. 启动 Accelerate 训练
# ==========================================
accelerate launch \
    --config_file /workspace/trl-main/examples/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes "$NUM_PROCESSES" \
    --mixed_precision bf16 \
    refGRPO_train.py \
    --abs_path /workspace/trl-main \
    --llm_size "$LLM_SIZE" \
    --llm_step "$LLM_STEP" \
    --sft_model_dir "$SFT_MODEL_DIR" \
    --judge_c \
    --judge_c_weight "$JUDGE_C_WEIGHT" \
    --re_exe_reward \
    --re_exe_reward_weight "$RE_EXE_REWARD_WEIGHT" \
    --syntax_reward \
    --syntax_reward_weight "$SYNTAX_REWARD_WEIGHT" \
    --semantic_reward \
    --semantic_reward_weight "$SEMANTIC_REWARD_WEIGHT" \
    --batch_size "$BATCH_SIZE" \
    --train_id "$TRAIN_ID" \
    $OPTIONAL_ARGS
