# inference_multinode/multinode.sh
#!/usr/bin/env bash
set -euo pipefail

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---------- Modules / Env (best-effort) ----------
module load gcc/15.1.0 || true
module load cuda/12.6   || true

# CUDA_HOME
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
export PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME:+$CUDA_HOME/lib64:}${LD_LIBRARY_PATH:-}"

# ---------- Rank env (OpenMPI / SLURM) ----------
RANK="${OMPI_COMM_WORLD_RANK:-${RANK:-${SLURM_PROCID:-0}}}"
WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:-${WORLD_SIZE:-${SLURM_NTASKS:-1}}}"
LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK:-${LOCAL_RANK:-${SLURM_LOCALID:-0}}}"

# ---------- Node-local caches ----------
NODE_LOCAL_BASE="${SLURM_TMPDIR:-/tmp/${USER}}"
mkdir -p "${NODE_LOCAL_BASE}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_DISABLE_MEMORY_MAPPING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${NODE_LOCAL_BASE}/prle_triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${NODE_LOCAL_BASE}/torchinductor}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# ---------- Rendezvous ----------
: "${SCRATCH:?SCRATCH must be set}"
CACHE_BASE="${SCRATCH%/}/prle_cache"
mkdir -p "${CACHE_BASE}"
MASTER_FILE="${CACHE_BASE}/master_${SLURM_JOB_ID:-$$}.addr"

if [[ "$RANK" == "0" ]]; then
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1 > "$MASTER_FILE"
  else
    hostname -s > "$MASTER_FILE"
  fi
fi

for _ in $(seq 1 120); do [[ -s "$MASTER_FILE" ]] && break; sleep 1; done
if [[ ! -s "$MASTER_FILE" ]]; then
  echo "Failed to discover MASTER_ADDR in time (file: $MASTER_FILE)" >&2
  exit 1
fi

export MASTER_ADDR="$(cat "$MASTER_FILE")"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "[launcher] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} WORLD_SIZE=${WORLD_SIZE} RANK=${RANK} LOCAL_RANK=${LOCAL_RANK}"

# ---------- Rank 0 log (mirror embeddings behavior) ----------
JOB_ID="${SLURM_JOB_ID:-$$}"
CACHE_BASE="${SCRATCH%/}/prle_cache"
LOG_DIR="${CACHE_BASE}/logs/${JOB_ID}"
mkdir -p "${LOG_DIR}"

if [[ "${RANK}" == "0" ]]; then
  export RANK0_LOG="${LOG_DIR}/rank0.log"
  echo "[launcher] rank0 will log progress to: ${RANK0_LOG}"
else
  unset RANK0_LOG
fi

# ---------- Run (allocate a PTY so tqdm renders live) ----------
if command -v script >/dev/null 2>&1; then
  exec script -q -c "python -u -m $1 ${*:2}" /dev/null
else
  exec python -u -m "$@"
fi
