#!/usr/bin/env bash
# embedding_multinode/multinode.sh
set -euo pipefail

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---------- Modules / Env ----------
module load gcc/15.1.0
# Match your torch.version.cuda (you reported 12.4)
module load cuda/12.6

# (Optional) venv/conda
# source "${PROJECT_ROOT}/.venv/bin/activate" || true
# conda activate PRLE || true

# CUDA_HOME (derive from nvcc if module didn't set it)
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [[ -z "${CUDA_HOME:-}" || ! -d "$CUDA_HOME" ]]; then
  echo "ERROR: CUDA_HOME not set/valid. Load a CUDA module (e.g. cuda/12.4)." >&2
  exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# ---------- Rank env (OpenMPI / SLURM) ----------
RANK="${OMPI_COMM_WORLD_RANK:-${RANK:-${SLURM_PROCID:-0}}}"
WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:-${WORLD_SIZE:-${SLURM_NTASKS:-1}}}"
LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK:-${LOCAL_RANK:-${SLURM_LOCALID:-0}}}"

# ---------- Scratch (assume $SCRATCH is correct) ----------
: "${SCRATCH:?SCRATCH must be set to your scratch directory}"
PRLE_CACHE_BASE="${SCRATCH%/}/prle_cache"
mkdir -p "$PRLE_CACHE_BASE"


export DEEPSPEED_DISABLE_STRICT_BATCH_ASSERTS=1

# Node-local tmp to avoid NFS warnings for Triton + tmp
NODE_LOCAL_BASE="${SLURM_TMPDIR:-/tmp/$USER}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${NODE_LOCAL_BASE}/prle_triton}"
export TMPDIR="${TMPDIR:-${NODE_LOCAL_BASE}/prle_tmp}"
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"

# Shared-ish caches under $SCRATCH
export HF_HOME="${HF_HOME:-${PRLE_CACHE_BASE}/hf}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PRLE_CACHE_BASE}/torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PRLE_CACHE_BASE}/xdg}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$HF_HOME" "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME"

# ---------- Rendezvous ----------
JOB_ID="${SLURM_JOB_ID:-$$}"
MASTER_DIR="${PRLE_CACHE_BASE}/master_addrs"
mkdir -p "$MASTER_DIR"
MASTER_FILE="${MASTER_DIR}/${JOB_ID}.addr"

if [[ "$RANK" == "0" ]]; then
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1 > "$MASTER_FILE"
  else
    hostname -s > "$MASTER_FILE"
  fi
fi

# Wait for master file (max ~120s)
for _ in $(seq 1 120); do
  [[ -s "$MASTER_FILE" ]] && break
  sleep 1
done
if [[ ! -s "$MASTER_FILE" ]]; then
  echo "Failed to discover MASTER_ADDR in time (file: $MASTER_FILE)" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR="$(cat "$MASTER_FILE")"
export MASTER_PORT="${MASTER_PORT:-29500}"

# ---------- NCCL / Threads ----------
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# export NCCL_IB_DISABLE=1  # uncomment to force TCP for troubleshooting
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

echo "[launcher] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} WORLD_SIZE=${WORLD_SIZE} RANK=${RANK} LOCAL_RANK=${LOCAL_RANK}"
echo "[launcher] CUDA_HOME=${CUDA_HOME}"
echo "[launcher] SCRATCH=${SCRATCH}"
echo "[launcher] PRLE_CACHE_BASE=${PRLE_CACHE_BASE}"
echo "[launcher] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "[launcher] HF_HOME=${HF_HOME}"

# ---------- Run ----------
# Usage: multinode.sh path/to/config.yaml
exec python -m embedding_multinode.embed_dm_main --config "${1:?Missing config path}"
