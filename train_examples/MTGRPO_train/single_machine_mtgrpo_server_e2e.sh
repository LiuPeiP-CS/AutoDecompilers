#!/bin/bash

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NVB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export GLOO_SOCKET_IFNAME=eth0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export GLOO_TIMEOUT_SECONDS=7200

export VLLM_MODE_HINT="server"
export REWARD_MAX_WORKERS=${reward_max_workers:-140}
export REWARD_TIMEOUT=${reward_timeout:-250}
DDP_TIMEOUT=${ddp_timeout:-10800}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VLLM_HOST=${vllm_host:-"XXX.XXX.XXX.XXX"}     
VLLM_PORT=${vllm_port:-6745}
VLLM_GROUP_PORT=${vllm_group_port:-51216}


NUM_PROCESSES=${num_processes:-8}       
LLM_SIZE=${llm_size:-"6.7"}
LLM_STEP=${llm_step:-"4500"}
PROGRESS_ALPHA=${progress_alpha:-"0.15"}
BATCH_SIZE=${batch_size:-8}
DEC_MODE=${dec_mode:-"e2e"}
DATASET_PATH=${dataset_path:-"rl4096train4end2end.jsonl"}
NUM_ITERATIONS=${num_iterations:-2}
TRAIN_ID=${train_id:-3}
MAX_TURNS=${max_turns:-3}
ONLY_FINAL_TURN_REWARD=${only_final_turn_reward:-"false"}
SAME_ADV=${same_adv:-"true"}

JUDGE_C_WEIGHT=${judge_c_weight:-0.5}
RE_EXE_REWARD_WEIGHT=${re_exe_reward_weight:-1.0}
SYNTAX_REWARD_WEIGHT=${syntax_reward_weight:-0.3}
SEMANTIC_REWARD_WEIGHT=${semantic_reward_weight:-0.2}

SFT_MODEL_DIR="/path/to/sft/checkpoints/SFT-${LLM_SIZE}B-FullSFT-checkpoints/checkpoint-${LLM_STEP}"

OPTIONAL_ARGS=""
[ "$ONLY_FINAL_TURN_REWARD" = "true" ] && OPTIONAL_ARGS="$OPTIONAL_ARGS --only_final_turn_reward"
[ "$SAME_ADV" = "true" ]               && OPTIONAL_ARGS="$OPTIONAL_ARGS --same_adv"

echo "=== 🚀 server mode ==="
echo "Number of training processes : $NUM_PROCESSES"
echo "vLLM service                 : $VLLM_HOST:$VLLM_PORT (group_port=$VLLM_GROUP_PORT)"
echo "SFT model                    : $SFT_MODEL_DIR"
echo "==============================="

accelerate launch \
    --config_file /private/trl-main/trl/accelerate_configs/zero3.yaml \
    --num_processes "$NUM_PROCESSES" \
    MTGRPO_train_server.py \
    --abs_path /private/trl-main \
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
    --grad_accum_steps 8 \
    --save_model_ratio 0.1 \
    --beta 0.02 \
    --vllm_tensor_parallel_size 1 \
    --dec_mode "$DEC_MODE" \
    --dataset_path "$DATASET_PATH" \
    --num_iterations "$NUM_ITERATIONS" \
    --train_id "$TRAIN_ID" \
    --vllm_mode server \
    --vllm_server_host "$VLLM_HOST" \
    --vllm_server_port "$VLLM_PORT" \
    --vllm_group_port "$VLLM_GROUP_PORT" \
    --ddp_timeout "$DDP_TIMEOUT" \
    $OPTIONAL_ARGS 2>&1 | tee single_machine_mtgrpo_server_e2e.log

# vllm_host=vllm_ip num_processes=8 bash single_machine_mtgrpo_server_e2e.sh


 ## attention !!!
 # we must revise the "TOKENIZER_NAME" in trl/envs/multiturn_env_server.py, and make sure it is the same model as SFT_MODEL_DIR
 # decrease the batch_size, and should increase the grad_accum_steps correspondingly