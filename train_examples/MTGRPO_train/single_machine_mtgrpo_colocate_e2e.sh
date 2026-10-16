#!/bin/bash

NUM_PROCESSES=${num_processes:-8}
LLM_SIZE=${llm_size:-"6.7"}
LLM_STEP=${llm_step:-"1500"}
PROGRESS_ALPHA=${progress_alpha:-"0.15"}
BATCH_SIZE=${batch_size:-8}
DEC_MODE=${dec_mode:-"e2e"}        
DATASET_PATH=${dataset_path:-"rl4096train4end2end.jsonl"}
NUM_ITERATIONS=${num_iterations:-2}   
TRAIN_ID=${train_id:-0}
MAX_TURNS=${max_turns:-3}


ONLY_FINAL_TURN_REWARD=${only_final_turn_reward:-"false"}
SAME_ADV=${same_adv:-"true"}

JUDGE_C_WEIGHT=${judge_c_weight:-0.5}
RE_EXE_REWARD_WEIGHT=${re_exe_reward_weight:-1.0}
SYNTAX_REWARD_WEIGHT=${syntax_reward_weight:-0.3}
SEMANTIC_REWARD_WEIGHT=${semantic_reward_weight:-0.2}

SFT_MODEL_DIR="/path/to/sft/checkpoints/SFT-${LLM_SIZE}B-FullSFT-checkpoints/checkpoint-${LLM_STEP}"


OPTIONAL_ARGS=""

if [ "$ONLY_FINAL_TURN_REWARD" = "true" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --only_final_turn_reward"
    echo "🔍 Status: only_final_turn_reward enabled"
else
    echo "🔍 Status: only_final_turn_reward disabled"
fi

# Handle same_adv
if [ "$SAME_ADV" = "true" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --same_adv"
    echo "🔍 Status: same_adv enabled"
else
    echo "🔍 Status: same_adv disabled"
fi


echo "=== 🚀 MTGRPO Training Configuration ==="
echo "Number of processes (num_processes) : $NUM_PROCESSES"
echo "Model size (llm_size)               : $LLM_SIZE"
echo "Model step (llm_step)               : $LLM_STEP"
echo "SFT model path                      : $SFT_MODEL_DIR"
echo "Progress Alpha                      : $PROGRESS_ALPHA"
echo "Per-GPU Batch Size                  : $BATCH_SIZE"
echo "Dataset mode (dec_mode)             : $DEC_MODE"
echo "Dataset path (dataset_path)         : $DATASET_PATH"
echo "Number of epochs                    : $NUM_ITERATIONS"
echo "==============================="


accelerate launch \
    --config_file /workspace/trl-main/examples/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes "$NUM_PROCESSES" \
    MTGRPO_train_colocate.py \
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
    --max_turns "$MAX_TURNS" \
    --progress_alpha "$PROGRESS_ALPHA" \
    --num_generations 4 \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --grad_accum_steps 4 \
    --save_model_ratio 0.1 \
    --beta 0.02 \
    --vllm_tensor_parallel_size 1 \
    --dec_mode "$DEC_MODE" \
    --dataset_path "$DATASET_PATH" \
    --num_iterations "$NUM_ITERATIONS" \
    --train_id "$TRAIN_ID" \
    $OPTIONAL_ARGS


