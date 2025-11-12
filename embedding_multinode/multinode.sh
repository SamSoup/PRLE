# embedding_multinode/multinode.sh
#!/usr/bin/env bash
set -euo pipefail

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---------- Modules / Env ----------
module load gcc/15.1.0 || true
module load cuda/12.6 || true

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
if [[ -z "${CUDA_HOME:-}" || ! -d "$CUDA_HOME" ]]; then
  echo "ERROR: CUDA_HOME not set/valid." >&2
  exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# ---------- Rank env (OpenMPI / SLURM) ----------
RANK="${OMPI_COMM_WORLD_RANK:-${RANK:-${SLURM_PROCID:-0}}}"
WORLD_SIZE="${OMPI_COMM_WORLD_SIZE:-${WORLD_SIZE:-${SLURM_NTASKS:-1}}}"
LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK:-${LOCAL_RANK:-${SLURM_LOCALID:-0}}}"

# ---------- Scratch ----------
: "${SCRATCH:?SCRATCH must be set to your scratch directory}"
CACHE_BASE="${SCRATCH%/}/prle_cache"
mkdir -p "$CACHE_BASE"

export HF_HOME="${HF_HOME:-${CACHE_BASE}/hf}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_BASE}/torch_extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_BASE}/xdg}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

# Put datasets cache on node-local/scratch (avoid NFS + mmap SIGBUS)
export HF_DATASETS_CACHE="${CACHE_BASE}/hf_datasets"     # e.g. /scratch/.../prle_cache/hf_datasets
mkdir -p "${HF_DATASETS_CACHE}"

# Absolutely kill memory mapping and tokenizers forks
export HF_DATASETS_DISABLE_MEMORY_MAPPING=1
export TOKENIZERS_PARALLELISM=false

# Avoid Datasets multiprocessing writing via shared memory accidents
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0  # leave default behavior; we’ll force keep_in_memory in code

# --- DeepSpeed Triton autotune cache on node-local storage (avoid NFS) ---
# Prefer SLURM_TMPDIR if available; otherwise /tmp/$USER
NODE_LOCAL_BASE="${SLURM_TMPDIR:-/tmp/${USER}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${NODE_LOCAL_BASE}/prle_triton}"
mkdir -p "${TRITON_CACHE_DIR}"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${NODE_LOCAL_BASE}/torchinductor}"
export XLA_CACHE_DIR="${XLA_CACHE_DIR:-${NODE_LOCAL_BASE}/xla_cache}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${XLA_CACHE_DIR}"

# ---------- Rendezvous ----------
JOB_ID="${SLURM_JOB_ID:-$$}"
MASTER_FILE="${CACHE_BASE}/master_${JOB_ID}.addr"
mkdir -p "$(dirname "$MASTER_FILE")"

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

# ---------- NCCL / Threads ----------
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

# more graceful NCCL/Store shutdown (avoids extra chatter on exit)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0    # was 1; disable during teardown
export NCCL_SHM_DISABLE=1                   # avoid SYSV shm corner cases


echo "[launcher] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} WORLD_SIZE=${WORLD_SIZE} RANK=${RANK} LOCAL_RANK=${LOCAL_RANK}"
echo "[launcher] CUDA_HOME=${CUDA_HOME}"
echo "[launcher] SCRATCH=${SCRATCH}"
echo "[launcher] HF_HOME=${HF_HOME}"

# ---------- Run ----------
# exec python -m embedding_multinode.embed_dm_main --config "${1:?Missing config path}"

# ---------- Rank 0 log ----------
LOG_DIR="${CACHE_BASE}/logs/${JOB_ID}"
mkdir -p "${LOG_DIR}"
if [[ "${RANK}" == "0" ]]; then
  export RANK0_LOG="${LOG_DIR}/rank0.log"
  echo "[launcher] rank0 will log progress to: ${RANK0_LOG}"
else
  unset RANK0_LOG
fi

# ---------- Run ----------
export HF_HUB_DISABLE_PROGRESS_BARS=0
exec python -u -m embedding_multinode.embed_dm_main --config "${1:?Missing config path}"