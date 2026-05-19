#!/usr/bin/env bash
#
# Run lm_eval suite against llama-server instances started automatically.
# llama-server should be from the fork, so that it is compatible with lm-eval (https://github.com/Ni-co-la-s/llama.cpp-gemmeh)
#
# Example:
# ./eval_gguf.sh \
#   --llama-cpp /home/me/llama.cpp \
#   --model /models/foo.gguf foo \
#   --model /models/bar.gguf bar
#
# Each --model takes:
#   --model <model_path> <model_name>

set -euo pipefail

# Defaults

HOST="127.0.0.1"
PORT=8000

NUM_CONCURRENT=1
CTX_SIZE="${CTX_SIZE:-2048}"
N_GPU_LAYERS=99

WANDB_PROJECT="${WANDB_PROJECT:-gemmeh-gguf}"

# Args

LLAMA_CPP_DIR=""
MODEL_PATHS=()
MODEL_NAMES=()

usage() {
  cat <<EOF
Usage:
  $0 --llama-cpp <path> --model <model_path> <model_name> [--model ...]

Example:
  $0 \
    --llama-cpp /home/me/llama.cpp \
    --model /models/foo.gguf foo \
    --model /models/bar.gguf bar
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llama-cpp)
      LLAMA_CPP_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL_PATHS+=("$2")
      MODEL_NAMES+=("$3")
      shift 3
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$LLAMA_CPP_DIR" ]]; then
  echo "ERROR: --llama-cpp is required"
  exit 1
fi

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
  echo "ERROR: at least one --model must be provided"
  exit 1
fi

LLAMA_SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "ERROR: llama-server not found or not executable:"
  echo "  $LLAMA_SERVER"
  exit 1
fi

mkdir -p tmp_results

# Eval function

run_eval_suite() {
  local MODEL_ALIAS="$1"

  local BASE_URL="http://${HOST}:${PORT}/v1/completions"

  local COMMON_ARGS="--model local-completions \
    --model_args model=${MODEL_ALIAS},base_url=${BASE_URL},num_concurrent=${NUM_CONCURRENT},max_length=${CTX_SIZE},tokenized_requests=False \
    --batch_size 1"

  local WANDB="project=${WANDB_PROJECT}"

  echo
  echo "Running eval suite for model: ${MODEL_ALIAS}"
  echo

  local OUT_DIR="tmp_results/${MODEL_ALIAS}"
  mkdir -p "${OUT_DIR}"

  lm_eval $COMMON_ARGS --tasks piqa           --num_fewshot 0  --output_path "${OUT_DIR}/piqa_0shot"            --wandb_args "$WANDB,name=${MODEL_ALIAS}_piqa_0shot"
  lm_eval $COMMON_ARGS --tasks winogrande     --num_fewshot 5  --output_path "${OUT_DIR}/winogrande_5shot"      --wandb_args "$WANDB,name=${MODEL_ALIAS}_winogrande_5shot"
  lm_eval $COMMON_ARGS --tasks arc_easy       --num_fewshot 0  --output_path "${OUT_DIR}/arc_easy_0shot"        --wandb_args "$WANDB,name=${MODEL_ALIAS}_arc_easy_0shot"
  lm_eval $COMMON_ARGS --tasks arc_challenge  --num_fewshot 25 --output_path "${OUT_DIR}/arc_challenge_25shot"  --wandb_args "$WANDB,name=${MODEL_ALIAS}_arc_challenge_25shot"
  lm_eval $COMMON_ARGS --tasks truthfulqa_mc2 --num_fewshot 0  --output_path "${OUT_DIR}/truthfulqa_mc2_0shot"  --wandb_args "$WANDB,name=${MODEL_ALIAS}_truthfulqa_mc2_0shot"
  echo
  echo "Finished eval suite for ${MODEL_ALIAS}"
  echo
}

# Main loop

for i in "${!MODEL_PATHS[@]}"; do
  MODEL_PATH="${MODEL_PATHS[$i]}"
  MODEL_NAME="${MODEL_NAMES[$i]}"

  echo
  echo "Starting llama-server for:"
  echo "  Model path : ${MODEL_PATH}"
  echo "  Model name : ${MODEL_NAME}"

  "${LLAMA_SERVER}" \
    --model "${MODEL_PATH}" \
    --alias "${MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --ctx-size "${CTX_SIZE}" \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --parallel 1 \
    --flash-attn on \
    > "tmp_results/${MODEL_NAME}_server.log" 2>&1 &

  SERVER_PID=$!

  cleanup() {
    if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "Stopping llama-server (${SERVER_PID})..."
      kill "${SERVER_PID}" || true
      wait "${SERVER_PID}" || true
    fi
  }

  trap cleanup EXIT

  echo "Waiting for server to become ready..."

  for attempt in {1..60}; do
    if curl -s "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "Server is up."
      break
    fi

    sleep 2

    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "ERROR: llama-server exited unexpectedly."
      exit 1
    fi

    if [[ $attempt -eq 60 ]]; then
      echo "ERROR: timeout waiting for llama-server."
      exit 1
    fi
  done

  run_eval_suite "${MODEL_NAME}"

  cleanup
  trap - EXIT

done

echo
echo "All evals completed."