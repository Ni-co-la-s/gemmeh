#!/usr/bin/env bash
#
# eval.sh — Start the server, run lm_eval suite, then shut it down.
# Usage 
#   src/gemmeh/pretrain/eval.sh checkpoints/path/to/best.pt  data/tokenizers/path/to/sentencepiece.model

set -euo pipefail

#  Config 
CHECKPOINT="${1}"
TOKENIZER="${2}"
PORT=8000
BASE_URL="http://localhost:${PORT}/v1/completions"

COMMON_ARGS="--model local-completions --model_args model=gemmeh,base_url=${BASE_URL},num_concurrent=1,tokenized_requests=False --batch_size 1"
WANDB='project=gemmeh-2024'

mkdir -p tmp_results

#  Start server 
echo "Starting server..."
pkill -f "server.py" 2>/dev/null || true
sleep 1

uv run -m gemmeh.pretrain.server \
    --checkpoint "$CHECKPOINT" \
    --tokenizer "$TOKENIZER" \
    --port $PORT \
    > tmp_results/server.log 2>&1 &
SERVER_PID=$!

#  Wait for server to be ready 
echo "Waiting for server (PID $SERVER_PID)..."
max_wait=120
waited=0
until curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; do
    sleep 2
    waited=$((waited + 2))
    if [ $waited -ge $max_wait ]; then
        echo "ERROR: Server did not start within ${max_wait}s. Check server.log."
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
done
echo "Server ready (took ${waited}s)"

#  Ensure server is killed on exit 
trap 'echo "Shutting down server..."; kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT


lm_eval $COMMON_ARGS --tasks hellaswag      --num_fewshot 10 --output_path tmp_results/hellaswag_10shot      --wandb_args "$WANDB,name=hellaswag_10shot" ;
lm_eval $COMMON_ARGS --tasks piqa           --num_fewshot 0  --output_path tmp_results/piqa_0shot            --wandb_args "$WANDB,name=piqa_0shot" ;
lm_eval $COMMON_ARGS --tasks winogrande     --num_fewshot 5  --output_path tmp_results/winogrande_5shot      --wandb_args "$WANDB,name=winogrande_5shot" ;
lm_eval $COMMON_ARGS --tasks arc_easy       --num_fewshot 0  --output_path tmp_results/arc_easy_0shot        --wandb_args "$WANDB,name=arc_easy_0shot" ;
lm_eval $COMMON_ARGS --tasks arc_challenge  --num_fewshot 25 --output_path tmp_results/arc_challenge_25shot  --wandb_args "$WANDB,name=arc_challenge_25shot" ;
lm_eval $COMMON_ARGS --tasks mmlu           --num_fewshot 5  --output_path tmp_results/mmlu_5shot            --wandb_args "$WANDB,name=mmlu_5shot" ;
lm_eval $COMMON_ARGS --tasks truthfulqa_mc2 --num_fewshot 0  --output_path tmp_results/truthfulqa_mc2_0shot  --wandb_args "$WANDB,name=truthfulqa_mc2_0shot" ;
lm_eval $COMMON_ARGS --tasks gpqa           --num_fewshot 5  --output_path tmp_results/gpqa_5shot            --wandb_args "$WANDB,name=gpqa_5shot" ;
lm_eval $COMMON_ARGS --tasks mmlu_pro       --num_fewshot 5  --output_path tmp_results/mmlu_pro_5shot        --wandb_args "$WANDB,name=mmlu_pro_5shot" ;
lm_eval $COMMON_ARGS --tasks gsm8k          --num_fewshot 4  --output_path tmp_results/gsm8k_4shot           --wandb_args "$WANDB,name=gsm8k_4shot" ;
lm_eval $COMMON_ARGS --tasks bbh            --num_fewshot 3  --output_path tmp_results/bbh_3shot             --wandb_args "$WANDB,name=bbh_3shot"

echo "All evals done."