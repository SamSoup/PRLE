#!/usr/bin/env bash
# Create a DeepSpeed Inference JSON manifest for a local HF model snapshot.
# Usage:
#   bash embedding_multinode/make_ds_manifest.sh MODEL_DIR OUT_JSON [-t TP_SIZE] [-d DTYPE]
# Example:
#   bash embedding_multinode/make_ds_manifest.sh \
#     /scratch/06782/ysu707/models/qwen3_next_80b_a3b_instr \
#     /scratch/06782/ysu707/models/qwen3_next_80b_a3b_instr/ds_inference_tp4_bf16.json \
#     -t 4 -d bf16

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MODEL_DIR OUT_JSON [-t TP_SIZE] [-d DTYPE]" >&2
  exit 1
fi

MODEL_DIR="$1"
OUT_JSON="$2"
shift 2

TP_SIZE=4
DTYPE=bf16

while getopts ":t:d:" opt; do
  case "$opt" in
    t) TP_SIZE="$OPTARG" ;;
    d) DTYPE="$OPTARG" ;;
    *) echo "Unknown flag: -$OPTARG" >&2; exit 2 ;;
  esac
done

# ---- Validate paths ---------------------------------------------------------
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: MODEL_DIR not found: $MODEL_DIR" >&2
  exit 1
fi
OUT_DIR="$(dirname "$OUT_JSON")"
mkdir -p "$OUT_DIR"

echo "[make_ds_manifest] MODEL_DIR=$MODEL_DIR"
echo "[make_ds_manifest] OUT_JSON=$OUT_JSON"
echo "[make_ds_manifest] TP_SIZE=$TP_SIZE  DTYPE=$DTYPE"

# Optional: move Triton cache off NFS to keep DS happy on some clusters
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/prle_triton}"
mkdir -p "$TRITON_CACHE_DIR"

# ---- Find a working converter module ---------------------------------------
candidates=(
  "deepspeed.ops.transformer.inference.ds_transformers_convert"
  "deepspeed.ops.transformer.inference.convert"
  "deepspeed.ops.inference.ds_transformers_convert"
)

found=""
for mod in "${candidates[@]}"; do
  if python - <<PY >/dev/null 2>&1
import importlib
import sys
try:
    importlib.import_module("$mod")
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
  then
    found="$mod"
    break
  fi
done

if [[ -z "$found" ]]; then
  echo "ERROR: Could not find a DeepSpeed HF→DS manifest converter module." >&2
  echo "Tried: ${candidates[*]}" >&2
  echo "Fix: upgrade DeepSpeed (e.g., pip install -U deepspeed) and retry." >&2
  exit 3
fi

echo "[make_ds_manifest] Using converter module: $found"

# ---- Run the converter ------------------------------------------------------
python -m "$found" \
  --checkpoint_dir "$MODEL_DIR" \
  --dtype "$DTYPE" \
  --tp_size "$TP_SIZE" \
  --output_file "$OUT_JSON"

echo "[make_ds_manifest] Wrote manifest: $OUT_JSON"
