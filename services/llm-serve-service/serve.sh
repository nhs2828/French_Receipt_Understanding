#!/bin/bash
set -euo pipefail

exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --quantization "$QUANTIZATION" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-prefix-caching \
  --host 0.0.0.0 \
  --port "$PORT" \
  ${EXTRA_ARGS:-}