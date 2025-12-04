import argparse
import atexit
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr
from tqdm.auto import tqdm

tqdm.monitor_interval = 0  # avoid tqdm background thread

from .big_infer_zero3 import DeepSpeedZero3LM
from .datasets import (
    make_module,
)  # expected: (dm, SYSTEM_MSG, build_user_message, parse_reply_to_0_1)


# ---------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------


@dataclass
class InferConfig:
    model: str
    out_dir: str

    dataset: str = "stsb"  # stsb | sickr_sts | sts22_crosslingual
    limit: Optional[int] = None
    icl_n: int = 0
    batch_size: int = 8

    max_new_tokens: int = 10
    temperature: float = 0.0

    dtype: str = "bfloat16"
    max_length: int = 512

    backend: str = "ds_zero3"
    model_local_dir: Optional[str] = None
    ds_config_path: Optional[str] = None
    cache_dir: Optional[str] = None

    heartbeat_every: int = 100

    # how long rank0 waits for per-rank files (seconds)
    gather_timeout_sec: int = 3600


class _Tee:
    """Mirror a stream to a log file."""

    def __init__(self, stream, filepath: str):
        self._stream = stream
        self._fp = open(
            filepath, "a", buffering=1, encoding="utf-8", errors="replace"
        )
        atexit.register(self.close)

    def write(self, data):
        self._stream.write(data)
        self._fp.write(data)
        self.flush()

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._fp.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def close(self):
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass


def _rank0_print(msg: str) -> None:
    if int(os.environ.get("RANK", 0)) == 0:
        print(msg, flush=True)


def _norm01_to_raw_for_dataset(dataset: str, x: float) -> float:
    if dataset == "stsb":
        return float(x)
    elif dataset == "sickr_sts":
        return float(x * 4.0 + 1.0)
    elif dataset == "sts22_crosslingual":
        return float(x * 3.0 + 1.0)
    else:
        return float(x)


def load_config(path: str) -> InferConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "model" not in raw or "out_dir" not in raw:
        raise ValueError("Config must include 'model' and 'out_dir'.")

    raw.setdefault("dataset", "stsb")
    raw.setdefault("limit", None)
    raw.setdefault("icl_n", 0)
    raw.setdefault("batch_size", 8)

    raw.setdefault("max_new_tokens", 10)
    raw.setdefault("temperature", 0.0)

    raw.setdefault("dtype", "bfloat16")
    raw.setdefault("max_length", 512)

    raw.setdefault("backend", "ds_zero3")
    raw.setdefault("model_local_dir", None)
    raw.setdefault(
        "ds_config_path", "inference_multinode/ds_zero3_inference.json"
    )
    raw.setdefault("cache_dir", None)

    raw.setdefault("heartbeat_every", 100)
    raw.setdefault("gather_timeout_sec", 3600)

    return InferConfig(**raw)


# ---------------------------------------------------------------------
# Rank / env helpers (no dist.init_process_group here)
# ---------------------------------------------------------------------


def ensure_dist_env() -> Tuple[int, int, int]:
    """
    Ensure RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    are set in os.environ so that dist.init_process_group(init_method='env://')
    inside DeepSpeedZero3LM works.

    IMPORTANT: This function does NOT call torch.distributed.*; it only
    manipulates environment variables and CUDA device.
    """
    # RANK
    rank = os.environ.get("RANK")
    if rank is None:
        rank = (
            os.getenv("OMPI_COMM_WORLD_RANK")
            or os.getenv("SLURM_PROCID")
            or "0"
        )
        os.environ["RANK"] = rank

    # WORLD_SIZE
    world = os.environ.get("WORLD_SIZE")
    if world is None:
        world = (
            os.getenv("OMPI_COMM_WORLD_SIZE")
            or os.getenv("SLURM_NTASKS")
            or "1"
        )
        os.environ["WORLD_SIZE"] = world

    # LOCAL_RANK
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        local_rank = (
            os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
            or os.getenv("SLURM_LOCALID")
            or "0"
        )
        os.environ["LOCAL_RANK"] = local_rank

    # MASTER_ADDR / MASTER_PORT
    # Your launcher already sets MASTER_ADDR/MASTER_PORT; we just add fallbacks.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    # Set CUDA device based on LOCAL_RANK if GPU is available
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(local_rank))
        except Exception as e:
            print(
                f"[env] failed to set cuda device to {local_rank}: {e}",
                flush=True,
            )

    return int(rank), int(world), int(local_rank)


def iter_with_progress(it, total: Optional[int], desc: str, rank: int):
    pbar = tqdm(
        total=total,
        desc=desc,
        disable=(rank != 0),
        dynamic_ncols=False,
        leave=True,
        mininterval=0.1,
        smoothing=0.1,
        file=sys.stderr,
        ascii=True,
        ncols=100,
        maxinterval=1.0,
    )
    try:
        for idx, item in enumerate(it):
            yield idx, item
            if rank == 0:
                pbar.update(1)
    finally:
        if rank == 0:
            pbar.close()


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "sentence1",
                "sentence2",
                "gold_similarity_raw",
                "llm_similarity_raw",
                "gold_similarity_0_1",
                "llm_similarity_0_1",
                "raw_model_reply",
            ],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _shard_indices_for_rank(n: int, rank: int, world: int) -> List[int]:
    """
    Simple modulo-based sharding of indices [0..n-1] across ranks.
    Works even if world > n or sharding is uneven.
    """
    return [i for i in range(n) if (i % world) == rank]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    # ---- process setup ----
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # ---- config ----
    cfg = (
        load_config(sys.argv[sys.argv.index("--config") + 1])
        if "--config" in sys.argv
        else None
    )
    if cfg is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", required=True)
        args = ap.parse_args()
        cfg = load_config(args.config)

    # ---- ensure env for env:// and get rank/world ----
    rank, world, local_rank = ensure_dist_env()

    # ---- rank 0 logging ----
    if rank == 0:
        log_path = os.environ.get("RANK0_LOG")
        if log_path:
            sys.stdout = _Tee(sys.stdout, log_path)
            sys.stderr = _Tee(sys.stderr, log_path)

    # ---- config sanity checks ----
    if cfg.backend.lower() != "ds_zero3":
        raise ValueError("Only 'ds_zero3' backend implemented in this script.")
    if not cfg.model_local_dir or not os.path.isdir(cfg.model_local_dir):
        raise FileNotFoundError(
            f"model_local_dir not found: {cfg.model_local_dir}"
        )
    if not cfg.ds_config_path or not os.path.isfile(cfg.ds_config_path):
        raise FileNotFoundError(
            f"ds_config_path not found: {cfg.ds_config_path}"
        )

    _rank0_print(f"[world={world} local_rank={local_rank}]")
    _rank0_print(f"Config: {cfg}")

    # ---- dataset setup ----
    dm, SYSTEM_MSG, build_user_message, parse_reply_to_0_1 = make_module(
        cfg.dataset, limit=cfg.limit, seed=42
    )
    test_data, gold_scores, icl_examples = dm.prepare(icl_n=cfg.icl_n)
    total_examples = len(test_data)
    if rank == 0:
        print(f"[data] total examples: {total_examples}", flush=True)

    # ---- model init & warmup (DeepSpeed handles its own PG) ----
    ok = True
    try:
        lm = DeepSpeedZero3LM(
            model_id=cfg.model,
            model_local_dir=cfg.model_local_dir,
            dtype=cfg.dtype,
            max_length=cfg.max_length,
            ds_config_path=cfg.ds_config_path,
            cache_dir=cfg.cache_dir,
            trust_remote_code=True,
        )
        if rank == 0:
            print("[warmup] starting tiny generate...", flush=True)
        lm.warmup_generate(max_new_tokens=min(8, cfg.max_new_tokens))
        if rank == 0:
            print("[warmup] generate ok", flush=True)
    except Exception as e:
        print(f"[rank{rank}] warmup failed: {e}", flush=True)
        ok = False

    if not ok:
        # If warmup fails on a rank, just exit; no collectives to coordinate.
        return

    # ---- sharding across ranks ----
    my_indices = _shard_indices_for_rank(total_examples, rank=rank, world=world)
    my_pairs: List[Tuple[int, Any]] = [
        (idx, test_data[idx]) for idx in my_indices
    ]

    if rank == 0:
        _rank0_print(
            f"Sharded dataset across {world} ranks; "
            f"rank0 sees {len(my_pairs)} examples (total={total_examples})."
        )
    print(f"[rank{rank}] shard size: {len(my_pairs)}", flush=True)

    os.makedirs(cfg.out_dir, exist_ok=True)
    tmp_dir = os.path.join(cfg.out_dir, "rank_shards")
    os.makedirs(tmp_dir, exist_ok=True)

    rows_for_dump: List[Dict[str, Any]] = []
    my_preds: List[float] = []
    my_golds: List[float] = []

    B = max(1, int(cfg.batch_size))
    total_batches = (len(my_pairs) + B - 1) // B
    batches_done = 0

    # ---- main inference loop (NO torch.distributed calls here) ----
    try:
        for b_idx, _ in iter_with_progress(
            range(total_batches),
            total=total_batches,
            desc="[infer-test]",
            rank=rank,
        ):
            start = b_idx * B
            end = min(len(my_pairs), start + B)
            batch_pairs = my_pairs[start:end]
            if not batch_pairs:
                continue

            batches_done += 1

            sys_msgs = [SYSTEM_MSG] * len(batch_pairs)
            user_msgs: List[str] = []
            gold01_batch: List[float] = []

            for global_idx, (s1, s2, gold01) in batch_pairs:
                user_msgs.append(build_user_message(s1, s2, icl_examples))
                gold01_batch.append(float(gold01))

            try:
                replies = lm.batch_chat(
                    system_messages=sys_msgs,
                    user_messages=user_msgs,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                )
            except Exception as e:
                print(
                    f"[rank{rank}] [error] batch generation failed at batch {b_idx}: {e}",
                    flush=True,
                )
                ok = False
                break

            for (global_idx, (s1, s2, gold01)), raw in zip(
                batch_pairs, replies
            ):
                pred01 = float(parse_reply_to_0_1(raw))
                pred_raw = _norm01_to_raw_for_dataset(cfg.dataset, pred01)
                gold01_f = float(gold01)
                gold_raw = _norm01_to_raw_for_dataset(cfg.dataset, gold01_f)

                my_preds.append(pred01)
                my_golds.append(gold01_f)
                rows_for_dump.append(
                    {
                        "idx": int(global_idx),
                        "sentence1": s1,
                        "sentence2": s2,
                        "gold_similarity_raw": gold_raw,
                        "llm_similarity_raw": pred_raw,
                        "gold_similarity_0_1": gold01_f,
                        "llm_similarity_0_1": float(pred01),
                        "raw_model_reply": raw,
                    }
                )

            if ((b_idx + 1) % max(1, cfg.heartbeat_every)) == 0:
                print(
                    f"[hb][rank{rank}] {b_idx + 1}/{total_batches}", flush=True
                )
                time.sleep(0.05)
    except Exception as e:
        print(f"[rank{rank}] exception in loop: {e}", flush=True)
        ok = False

    # ---- flush CUDA (still no collectives) ----
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as e:
        print(f"[rank{rank}] cuda sync failed: {e}", flush=True)
        ok = False

    # ---- write per-rank shard file ----
    shard_path = os.path.join(tmp_dir, f"rank_{rank:05d}.json")
    shard_payload = {
        "rank": rank,
        "world_size": world,
        "rows": rows_for_dump,
        "preds01": my_preds,
        "golds01": my_golds,
        "ok": ok,
    }
    with open(shard_path, "w", encoding="utf-8") as f:
        json.dump(shard_payload, f)

    print(
        f"[rank{rank}] wrote shard file with {len(rows_for_dump)} rows "
        f"and {len(my_preds)} preds to {shard_path}",
        flush=True,
    )

    # ---- non-zero ranks: we're done ----
    if rank != 0:
        return

    # -----------------------------------------------------------------
    # Rank 0 only: wait for all shard files, then merge and compute metrics
    # -----------------------------------------------------------------
    print(
        f"[rank0] waiting for {world} shard files in {tmp_dir} "
        f"(timeout {cfg.gather_timeout_sec}s)...",
        flush=True,
    )
    deadline = time.time() + cfg.gather_timeout_sec
    expected_files = {f"rank_{r:05d}.json" for r in range(world)}

    while True:
        existing = set(os.listdir(tmp_dir))
        missing = expected_files - existing
        if not missing:
            break
        if time.time() > deadline:
            print(
                f"[rank0] timeout while waiting for shard files; "
                f"missing: {sorted(missing)}",
                flush=True,
            )
            break
        time.sleep(2.0)

    # Load whatever shard files we have
    all_rows: List[Dict[str, Any]] = []
    all_preds01_list: List[float] = []
    all_golds01_list: List[float] = []

    for fname in sorted(os.listdir(tmp_dir)):
        if not fname.startswith("rank_") or not fname.endswith(".json"):
            continue
        path = os.path.join(tmp_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"[rank0] failed to read shard {path}: {e}", flush=True)
            continue

        if not payload.get("ok", True):
            print(
                f"[rank0] shard {path} reported ok=False; skipping its data.",
                flush=True,
            )
            continue

        all_rows.extend(payload.get("rows", []))
        all_preds01_list.extend(payload.get("preds01", []))
        all_golds01_list.extend(payload.get("golds01", []))

    all_preds01 = np.array(all_preds01_list, dtype=float)
    all_golds01 = np.array(all_golds01_list, dtype=float)

    if all_preds01.size != total_examples:
        print(
            f"[rank0][warning] gathered {all_preds01.size} predictions but "
            f"dataset had {total_examples} examples.",
            flush=True,
        )

    # Sort rows by global index and reindex 0..N-1 for readability
    all_rows_sorted = sorted(all_rows, key=lambda r: int(r["idx"]))
    for new_idx, row in enumerate(all_rows_sorted):
        row["idx"] = new_idx

    # Convert normalized scores to raw scores
    to_raw = lambda v: _norm01_to_raw_for_dataset(cfg.dataset, float(v))
    all_preds_raw = np.array([to_raw(v) for v in all_preds01], dtype=float)
    all_golds_raw = np.array([to_raw(v) for v in all_golds01], dtype=float)

    def mse(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return float(np.mean((a - b) ** 2))

    if all_preds01.size > 1 and all_golds01.size > 1:
        pearson_norm, _ = pearsonr(all_preds01, all_golds01)
        mse_norm = mse(all_preds01, all_golds01)
        pearson_raw, _ = pearsonr(all_preds_raw, all_golds_raw)
        mse_raw = mse(all_preds_raw, all_golds_raw)
    else:
        pearson_norm = pearson_raw = float("nan")
        mse_norm = mse_raw = float("nan")

    metrics = {
        "num_pairs_scored": int(all_preds01.size),
        "pearson_correlation_norm": float(pearson_norm),
        "mse_norm": float(mse_norm),
        "pearson_correlation_raw": float(pearson_raw),
        "mse_raw": float(mse_raw),
        "model_name": cfg.model,
        "dataset": cfg.dataset,
        "icl_n": int(cfg.icl_n),
        "max_new_tokens": int(cfg.max_new_tokens),
        "temperature": float(cfg.temperature),
        "batch_size": int(cfg.batch_size),
        "label_range": (
            "0..1"
            if cfg.dataset == "stsb"
            else (
                "1..5 (also normalized via (x-1)/4)"
                if cfg.dataset == "sickr_sts"
                else "1..4 (also normalized via (x-1)/3)"
            )
        ),
        "total_examples": int(total_examples),
        "world_size": int(world),
    }

    preds_path = os.path.join(cfg.out_dir, "predictions.csv")
    metrics_path = os.path.join(cfg.out_dir, "metrics.json")
    icl_path = os.path.join(cfg.out_dir, "icl_examples.json")

    write_csv(preds_path, all_rows_sorted)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(icl_path, "w", encoding="utf-8") as f:
        json.dump(icl_examples, f, indent=2)

    print(f"\n[rank0] Saved predictions to: {preds_path}")
    print(f"[rank0] Saved metrics to:     {metrics_path}")
    print(f"[rank0] Saved ICL examples to:{icl_path}")
    print(f"[rank0] Per-rank shards kept in: {tmp_dir}")


if __name__ == "__main__":
    main()
