#!/usr/bin/env bash
#
# rys_search.sh — Full RYS grid search over all (i, j) pairs.
#
# For each pair, starts the server with in-memory layer duplication,
# runs HellaSwag with a limit, logs to wandb, records score to CSV,
# then kills the server and moves on.
#
# Usage:
#   ./src/gemmeh/rys/rys_search.sh checkpoints/path/to/best.pt  data/tokenizers/path/to/sentencepiece.model

set -euo pipefail

#  Config 

PORT=8000
BASE_URL="http://localhost:${PORT}/v1/completions"
NUM_LAYERS=12
LIMIT=1000
CSV_FILE="rys_results.csv"
TMP_RESULTS="tmp_results"
WANDB_PROJECT="gemmeh-2024-rys"

CHECKPOINT=${1}
TOKENIZER=${2}
I_START=${3:-0}
I_END=${4:-$NUM_LAYERS}


#  Initialize CSV 
if [ ! -f "$CSV_FILE" ]; then
    echo "i,j,dup_size,total_layers,acc,acc_norm,time_seconds" > "$CSV_FILE"
    echo "Created $CSV_FILE"
fi

#  Helper: wait for server to be ready 
wait_for_server() {
    local max_wait=120
    local waited=0
    while ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            echo "ERROR: Server did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "Server ready (took ${waited}s)"
}

#  Helper: check if (i,j) already done 
already_done() {
    local check_i=$1
    local check_j=$2
    if [ -f "$CSV_FILE" ]; then
        grep -q "^${check_i},${check_j}," "$CSV_FILE" && return 0
    fi
    return 1
}

#  Run baseline first (no duplication) 
if ! already_done "base" "base"; then
    echo ""
    echo "============================================="
    echo "Running baseline (no duplication)"
    echo "============================================="
    start_time=$(date +%s)

    pkill -f "gemmeh.rys.server" 2>/dev/null || true
    sleep 1

    uv run -m gemmeh.rys.server \
        --checkpoint "$CHECKPOINT" \
        --tokenizer "$TOKENIZER" \
        --number_layers "$NUM_LAYERS" \
        --port $PORT\
        > /dev/null 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server; then
        echo "FAILED: server didn't start for baseline"
        kill $SERVER_PID 2>/dev/null || true
    else
        rm -rf "$TMP_RESULTS"

        lm_eval \
            --model local-completions \
            --model_args "model=gemmeh-1b-baseline,base_url=${BASE_URL},num_concurrent=1,tokenized_requests=False" \
            --tasks hellaswag \
            --batch_size 1 \
            --limit $LIMIT \
            --output_path "$TMP_RESULTS" \
            --wandb_args "project=${WANDB_PROJECT},name=baseline" \
            --log_samples \
            2>&1 | tail -5

        RESULTS_FILE=$(find "$TMP_RESULTS" -name "results_*.json" | head -1)
        if [ -n "$RESULTS_FILE" ]; then
            acc=$(python3 -c "import json; r=json.load(open('$RESULTS_FILE')); print(r['results']['hellaswag']['acc,none'])")
            acc_norm=$(python3 -c "import json; r=json.load(open('$RESULTS_FILE')); print(r['results']['hellaswag']['acc_norm,none'])")
        else
            acc="NA"
            acc_norm="NA"
        fi

        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "base,base,0,$NUM_LAYERS,$acc,$acc_norm,$elapsed" >> "$CSV_FILE"
        echo "  -> BASELINE acc=$acc  acc_norm=$acc_norm  (${elapsed}s)"

        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        rm -rf "$TMP_RESULTS"
    fi
fi

#  Count total pairs (j starts at i, all are valid single-layer dups and up) 
total=0
for ((i=I_START; i<I_END; i++)); do
    for ((j=i; j<NUM_LAYERS; j++)); do
        total=$((total + 1))
    done
done
echo "Total (i,j) pairs: $total (plus baseline)"

#  Main loop 
count=0
for ((i=I_START; i<I_END; i++)); do
    for ((j=i; j<NUM_LAYERS; j++)); do
        count=$((count + 1))
        dup_size=$((j - i + 1))
        total_layers=$((NUM_LAYERS + dup_size))

        if already_done $i $j; then
            echo "[$count/$total] i=$i j=$j — already done, skipping"
            continue
        fi

        echo ""
        echo "============================================="
        echo "[$count/$total] i=$i j=$j (dup=$dup_size, layers=$total_layers)"
        echo "============================================="
        start_time=$(date +%s)

        pkill -f "gemmeh.rys.server" 2>/dev/null || true
        sleep 1

        uv run -m gemmeh.rys.server \
            --checkpoint "$CHECKPOINT" \
            --tokenizer "$TOKENIZER" \
            --number_layers "$NUM_LAYERS" \
            --rys_i $i --rys_j $j \
            --port $PORT \
            > /dev/null 2>&1 &
        SERVER_PID=$!

        if ! wait_for_server; then
            echo "FAILED: server didn't start for i=$i j=$j"
            kill $SERVER_PID 2>/dev/null || true
            continue
        fi

        rm -rf "$TMP_RESULTS"

        MODEL_NAME="gemmeh-1b-rys-i${i}-j${j}"

        lm_eval \
            --model local-completions \
            --model_args "model=${MODEL_NAME},base_url=${BASE_URL},num_concurrent=1,tokenized_requests=False" \
            --tasks hellaswag \
            --batch_size 1 \
            --limit $LIMIT \
            --output_path "$TMP_RESULTS" \
            --wandb_args "project=${WANDB_PROJECT},name=i${i}_j${j}_dup${dup_size}" \
            --log_samples \
            2>&1 | tail -5

        RESULTS_FILE=$(find "$TMP_RESULTS" -name "results_*.json" | head -1)
        if [ -n "$RESULTS_FILE" ]; then
            acc=$(python3 -c "import json; r=json.load(open('$RESULTS_FILE')); print(r['results']['hellaswag']['acc,none'])")
            acc_norm=$(python3 -c "import json; r=json.load(open('$RESULTS_FILE')); print(r['results']['hellaswag']['acc_norm,none'])")
        else
            echo "WARNING: No results file found for i=$i j=$j"
            acc="NA"
            acc_norm="NA"
        fi

        end_time=$(date +%s)
        elapsed=$((end_time - start_time))

        echo "$i,$j,$dup_size,$total_layers,$acc,$acc_norm,$elapsed" >> "$CSV_FILE"
        echo "  -> acc=$acc  acc_norm=$acc_norm  (${elapsed}s)"

        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        rm -rf "$TMP_RESULTS"

    done
done

echo ""
echo "============================================="
echo "RYS search complete! Results in $CSV_FILE"
echo "============================================="