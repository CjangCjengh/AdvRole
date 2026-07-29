#!/bin/bash
# Launch a vLLM OpenAI-compatible server on auto-selected free GPUs.
# Usage: bash scripts/serve_model.sh <model_path> <served_name> <port> [tp] [gpu_spec]
# GPU_SPEC="0,1" pins GPUs; default "auto" picks the <tp> cards with the
# most free memory (>= 40GB).
set -euo pipefail

MODEL_PATH=${1:?model path required}
SERVED_NAME=${2:?served name required}
PORT=${3:?port required}
TP=${4:-1}
GPU_SPEC=${GPU_SPEC:-${5:-auto}}

PYTHON=${PYTHON:-python}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$GPU_SPEC" == "auto" ]]; then
  GPU_IDS=$(cd "$REPO_ROOT" && TP="$TP" "$PYTHON" -c "
import os
from advrole.gpu import select_free_gpus
ids = select_free_gpus(int(os.environ['TP']), min_free_mb=40000)
print(','.join(map(str, ids)))
")
  if [[ -z "$GPU_IDS" ]]; then echo "No free GPUs!"; exit 1; fi
else
  GPU_IDS=$GPU_SPEC
fi

echo "[serve] model=$MODEL_PATH name=$SERVED_NAME port=$PORT tp=$TP gpus=$GPU_IDS"
mkdir -p logs
CUDA_VISIBLE_DEVICES=$GPU_IDS "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.88 \
  --trust-remote-code \
  --disable-log-requests \
  2>&1 | tee "logs/server_${SERVED_NAME}.log"
