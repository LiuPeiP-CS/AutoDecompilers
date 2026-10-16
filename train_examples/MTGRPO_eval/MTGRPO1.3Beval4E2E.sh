#!/bin/bash

BASE_PATH="/XXX/trl-main/train_examples/MTGRPO_eval"
MODEL_PATH="/path/to/your/checkpoint"

DIRB="/XXX/trl-main/train_examples/MTGRPO_eval/all_models_eval_results/1.3-e2e"
DATA_PATH="/XXX/trl-main/datasets/test/TestHumanEnd2end.jsonl"
ALLMETRICS="/XXX/trl-main/train_examples/MTGRPO_eval/all_models_eval_results/1.3-e2e-metric.txt"


[ ! -d "$MODEL_PATH" ] && {
    echo "Error: MODEL_PATH not exists: $MODEL_PATH"
    exit 1
}

[ ! -f "$BASE_PATH/MTGRPO_eval.py" ] && {
    echo "Error: MTGRPO_eval.py not exists: $BASE_PATH/MTGRPO_eval.py"
    exit 1
}

[ ! -f "$BASE_PATH/recompile_rl.py" ] && {
    echo "Error: recompile_rl.py not exists: $BASE_PATH/recompile_rl.py"
    exit 1
}

[ ! -f "$DATA_PATH" ] && {
    echo "Error: DATA_PATH not exists: $DATA_PATH"
    exit 1
}


mkdir -p "$DIRB"
mkdir -p "$(dirname "$ALLMETRICS")"

cleanup() {
    echo "[CLEANUP] Killing residual gcc/compile processes..."

    pkill -KILL -f "gcc" 2>/dev/null
    pkill -KILL -f "combine" 2>/dev/null
    pkill -KILL -f "onlyfunc" 2>/dev/null
}

trap cleanup EXIT INT TERM


MODEL_NAME=$(basename "$MODEL_PATH")
AGENT_OUTDIR="$DIRB/$MODEL_NAME"

mkdir -p "$AGENT_OUTDIR"

echo "========================================"
echo "MODEL_NAME   : $MODEL_NAME"
echo "MODEL_PATH   : $MODEL_PATH"
echo "AGENT_OUTDIR : $AGENT_OUTDIR"
echo "========================================"


if ! python "$BASE_PATH/MTGRPO_eval.py" \
    --model_dir "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$AGENT_OUTDIR" \
    --tensor_parallel_size 8 \
    --dataset humaneval \
    --batch_size 8 \
    --max_turns 3
then
    echo "[ERROR] MTGRPO_eval.py failed for ${MODEL_NAME}"
    exit 1
fi

echo "[DONE] MTGRPO evaluation: $MODEL_NAME"


RESULT_FILE=$(find "$AGENT_OUTDIR" \
    -maxdepth 1 \
    -type f \
    -name "*.jsonl" \
    -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-)

if [ -z "$RESULT_FILE" ]; then
    echo "[ERROR] No JSONL result found in $AGENT_OUTDIR"
    exit 1
fi

echo "[INFO] RESULT_FILE: $RESULT_FILE"


setsid python "$BASE_PATH/recompile_rl.py" \
    -i "$RESULT_FILE" \
    -o "$ALLMETRICS" &

RECOMPILE_PID=$!

echo "[INFO] recompile_rl.py PID: $RECOMPILE_PID"

wait "$RECOMPILE_PID"
RECOMPILE_EXIT=$?


cleanup
trap - EXIT INT TERM


if [ "$RECOMPILE_EXIT" -ne 0 ]; then
    echo "[ERROR] recompile_rl.py failed for ${MODEL_NAME}"
    exit 1
fi

echo "========================================"
echo "All finished."
echo "MODEL : $MODEL_NAME"
echo "RESULT: $RESULT_FILE"
echo "METRIC: $ALLMETRICS"
echo "========================================"