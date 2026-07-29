#!/bin/bash
# End-to-end smoke test: 2 actor steps + 1 rewriter step + 16 rewrites.
# Small model by default (colocated FSDP+vLLM OOMs at 7B on 48GB cards
# during the weight-sync step); override with SMOKE_MODEL=... DATASET=...
# Requires an env with vllm+verl (activate it, or set ENV_PREFIX).
set -euo pipefail
cd "$(dirname "$0")/.."

if ! python -c "import vllm, verl" 2>/dev/null; then
  if [ -n "${ENV_PREFIX:-}" ] && [ -x "${ENV_PREFIX}/bin/python" ]; then
    export PATH="${ENV_PREFIX}/bin:${PATH}"
  else
    echo "ERROR: python on PATH lacks vllm/verl." >&2
    echo "Activate the env first, or export ENV_PREFIX=<conda prefix>." >&2
    exit 1
  fi
fi

DATASET=${DATASET:-charactereval}
SMOKE_MODEL=${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
SMOKE_DIR=outputs/smoke_$(date +%Y%m%d_%H%M%S)
echo "=== AdvRole smoke test: $DATASET -> $SMOKE_DIR (model: $SMOKE_MODEL) ==="

python train.py --dataset "$DATASET" --set \
  model.actor.base_model="$SMOKE_MODEL" \
  model.rewriter.base_model="$SMOKE_MODEL" \
  training.num_epochs=1 \
  training.actor_steps_per_epoch=2 \
  training.rewriter_steps_per_epoch=1 \
  grpo.batch_size=4 \
  grpo.group_size=4 \
  grpo.ppo_mini_batch_size=4 \
  grpo.ppo_micro_batch_size_per_gpu=1 \
  grpo.max_prompt_length=2048 \
  rewriter.num_actor_samples=1 \
  rewriter.augment_limit=16 \
  hardware.actor_gpus=auto:2 \
  output.output_dir="$SMOKE_DIR"

echo "=== smoke training done: $SMOKE_DIR ==="
