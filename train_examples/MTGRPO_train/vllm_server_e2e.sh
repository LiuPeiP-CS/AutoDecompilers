#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
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

LLM_SIZE=${llm_size:-"6.7"}
LLM_STEP=${llm_step:-"4500"}
SFT_MODEL_DIR=${sft_model_dir:-"/path/to/sft/checkpoints/SFT-${LLM_SIZE}B-FullSFT-checkpoints/checkpoint-${LLM_STEP}"}

HOST=${host:-"0.0.0.0"}
PORT=${port:-6745}

TP_SIZE=${tp_size:-1}       
DP_SIZE=${dp_size:-4}       
GPU_MEM_UTIL=${gpu_mem_util:-0.85}
MAX_MODEL_LEN=${max_model_len:-16384}


echo "=== 🚀 vLLM Server Configuration ==="
echo "Model path    : $SFT_MODEL_DIR"
echo "Listening on  : $HOST:$PORT"
echo "TP / DP       : $TP_SIZE / $DP_SIZE"
echo "Visible GPUs  : $CUDA_VISIBLE_DEVICES"
echo "==========================="

trl vllm-serve \
    --model "$SFT_MODEL_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor_parallel_size "$TP_SIZE" \
    --data_parallel_size "$DP_SIZE" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --max_model_len "$MAX_MODEL_LEN" \
    --dtype auto \
    2>&1 | tee vllm_server_e2e.log

# bash vllm_server.sh
