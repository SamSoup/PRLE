import yaml
import argparse, csv, json, os, sys, time, atexit
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
import pickle
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from scipy.stats import pearsonr
from tqdm.auto import tqdm

tqdm.monitor_interval = 0


from .big_infer_zero3 import DeepSpeedZero3LM
from .datasets import (
    make_module,
)  # registry loader (provides SYSTEM_MSG, build_user_message, parse_reply_to_0_1)


@dataclass
class InferConfig:
    model: str
    out_dir: str

    dataset: str = "stsb"  # stsb | sickr_sts | sts22_crosslingual
    limit: Optional[int] = None
    icl_n: int = 0
    batch_size: int = 8  # NEW: batched generation

    max_new_tokens: int = 10
    temperature: float = 0.0

    dtype: str = "bfloat16"
    max_length: int = 512

    backend: str = "ds_zero3"
    model_local_dir: Optional[str] = None
    ds_config_path: Optional[str] = None
    cache_dir: Optional[str] = None

    heartbeat_every: int = 100


class _Tee:
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


def _norm01_to_raw_for_dataset(dataset: str, x: float) -> float:
    # Inverse of the dataset’s normalization used in the dataset loaders.
    # stsb: gold in [0,1] already
    # sickr_sts: norm = (raw-1)/4  -> raw = norm*4+1  (raw in [1,5])
    # sts22_crosslingual: norm = (raw-1)/3  -> raw = norm*3+1  (raw in [1,4])
    if dataset == "stsb":
        return float(x)
    elif dataset == "sickr_sts":
        return float(x * 4.0 + 1.0)
    elif dataset == "sts22_crosslingual":
        return float(x * 3.0 + 1.0)
    else:
        # default: treat as already 0..1
        return float(x)


def any_rank_failed(local_ok: bool) -> bool:
    t = torch.tensor(
        [0 if local_ok else 1],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() > 0


def _rank0_print(msg: str) -> None:
    if int(os.environ.get("RANK", 0)) == 0:
        print(msg, flush=True)


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
    return InferConfig(**raw)


def init_dist() -> Tuple[int, int, int]:
    os.environ.setdefault("RANK", os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    os.environ.setdefault("WORLD_SIZE", os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    os.environ.setdefault(
        "LOCAL_RANK", os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0")
    )

    if "MASTER_ADDR" not in os.environ:
        raise RuntimeError(
            "MASTER_ADDR not set. Ensure your launcher exports MASTER_ADDR."
        )
    os.environ.setdefault("MASTER_PORT", "29500")

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        # device_id=local_rank silences the “device unknown” warning and prevents mismatch
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(minutes=60),
            init_method="env://",
            device_id=local_rank if torch.cuda.is_available() else None,
        )

    # device-aware barrier
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()

    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), local_rank


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
                # raw-scale (dataset native)
                "gold_similarity_raw",
                "llm_similarity_raw",
                # normalized 0..1
                "gold_similarity_0_1",
                "llm_similarity_0_1",
                "raw_model_reply",
            ],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    try:
        mp.set_start_method("spawn", force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

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

    rank, world, local_rank = init_dist()

    if rank == 0:
        log_path = os.environ.get("RANK0_LOG")
        if log_path:
            sys.stdout = _Tee(sys.stdout, log_path)
            sys.stderr = _Tee(sys.stderr, log_path)

    if cfg.backend.lower() != "ds_zero3":
        raise ValueError("Only 'ds_zero3' backend implemented.")
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

    # Load dataset via registry
    dm, SYSTEM_MSG, build_user_message, parse_reply_to_0_1 = make_module(
        cfg.dataset, limit=cfg.limit, seed=42
    )

    test_data, gold_scores, icl_examples = dm.prepare(icl_n=cfg.icl_n)

    # Model
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

        # >>> NEW: make sure all ranks finished DS init before any generates
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
        # <<<

        if rank == 0:
            print("[warmup] starting tiny generate...", flush=True)
        lm.warmup_generate(max_new_tokens=min(8, cfg.max_new_tokens))
        if rank == 0:
            print("[warmup] generate ok", flush=True)
    except Exception as e:
        if rank == 0:
            print(f"[warmup] skipped due to: {e}", flush=True)
        ok = False

    if any_rank_failed(ok):
        try:
            if dist.get_backend() == "nccl" and torch.cuda.is_available():
                dist.barrier(device_ids=[local_rank])
            else:
                dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
        sys.exit(1)

    # Partition across ranks
    my_items = [x for i, x in enumerate(test_data) if (i % world) == rank]
    _rank0_print(
        f"Split test set across ranks; rank0 sees ~{len(my_items)} items."
    )
    os.makedirs(cfg.out_dir, exist_ok=True)

    rows_for_dump: List[Dict[str, Any]] = []
    my_preds: List[float] = []
    my_golds: List[float] = []

    B = max(1, int(cfg.batch_size))
    total_batches = (len(my_items) + B - 1) // B

    for b_idx, _ in iter_with_progress(
        range(total_batches),
        total=total_batches,
        desc="[infer-test]",
        rank=rank,
    ):
        start = b_idx * B
        end = min(len(my_items), start + B)
        batch = my_items[start:end]  # list of (s1, s2, gold01)
        if not batch:
            continue

        # Build messages
        sys_msgs = [SYSTEM_MSG] * len(batch)
        user_msgs: List[str] = []
        gold01_batch: List[float] = []
        for s1, s2, gold01 in batch:
            user_msgs.append(build_user_message(s1, s2, icl_examples))
            gold01_batch.append(float(gold01))

        # if rank == 0:
        #     # log a few messages
        #     _rank0_print(
        #         f"Exampel User Messages: {user_msgs[0]}\n{user_msgs[1]}"
        #     )

        # Generate in batch
        try:
            replies = lm.batch_chat(
                system_messages=sys_msgs,
                user_messages=user_msgs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
            )
        except Exception as e:
            if rank == 0:
                print(
                    f"[error] batch generation failed at batch {b_idx}: {e}",
                    flush=True,
                )
            continue

        # Parse + record
        for j, (s1, s2, gold01) in enumerate(batch):
            raw = replies[j]
            pred01 = float(
                parse_reply_to_0_1(raw)
            )  # normalized [0,1] via dataset parser
            # Convert to dataset raw scale for reporting
            pred_raw = _norm01_to_raw_for_dataset(cfg.dataset, pred01)
            gold_raw = _norm01_to_raw_for_dataset(cfg.dataset, float(gold01))

            my_preds.append(pred01)
            my_golds.append(gold01)

            rows_for_dump.append(
                {
                    "idx": start + j,
                    "sentence1": s1,
                    "sentence2": s2,
                    "gold_similarity_raw": gold_raw,
                    "llm_similarity_raw": pred_raw,
                    "gold_similarity_0_1": float(gold01),
                    "llm_similarity_0_1": float(pred01),
                    "raw_model_reply": raw,
                }
            )

        # Heatbeat on all nodes because sharded dataset
        if ((b_idx + 1) % max(1, cfg.heartbeat_every)) == 0:
            print(f"[hb][rank{rank}] {b_idx + 1}/{total_batches}", flush=True)

            time.sleep(0.05)

    torch.cuda.synchronize()
    dist.barrier(
        device_ids=(
            [local_rank]
            if (dist.get_backend() == "nccl" and torch.cuda.is_available())
            else None
        )
    )
    print(f"[rank{rank}] finished compute; entering gather.", flush=True)
    print(
        f"[rank{rank}] starting gathers: rows={len(rows_for_dump)}, preds={len(my_preds)}, golds={len(my_golds)}",
        flush=True,
    )

    # -------- Gather to rank0 (NCCL-safe, tensor-based) --------
    def _gather_list_nccl_safe(py_list):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        payload = pickle.dumps(py_list, protocol=pickle.HIGHEST_PROTOCOL)

        np_buf = np.frombuffer(payload, dtype=np.uint8).copy()  # make writable
        local = torch.from_numpy(np_buf).to(device)

        # size first
        sz = torch.tensor([local.numel()], dtype=torch.int64, device=device)
        sizes = [torch.zeros_like(sz) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, sz)

        max_sz = int(torch.stack(sizes).max().item())
        pad = torch.zeros(max_sz, dtype=torch.uint8, device=device)
        pad[: local.numel()] = local

        bufs = [torch.empty_like(pad) for _ in range(dist.get_world_size())]
        dist.all_gather(bufs, pad)

        outs = []
        for i, b in enumerate(bufs):
            bl = int(sizes[i].item())
            outs.append(pickle.loads(bytes(b[:bl].tolist())))
        return outs

    gathered_rows = _gather_list_nccl_safe(rows_for_dump)
    gathered_preds = _gather_list_nccl_safe(my_preds)
    gathered_golds = _gather_list_nccl_safe(my_golds)

    if rank == 0:
        # flatten lists from ranks
        all_rows = [r for sub in gathered_rows for r in sub]
        all_preds01 = np.array(
            [x for sub in gathered_preds for x in sub], dtype=float
        )  # normalized
        all_golds01 = np.array(
            [x for sub in gathered_golds for x in sub], dtype=float
        )  # normalized

        # Map to raw scale using the same inverse
        to_raw = lambda v: _norm01_to_raw_for_dataset(cfg.dataset, float(v))
        all_preds_raw = np.array([to_raw(v) for v in all_preds01], dtype=float)
        all_golds_raw = np.array([to_raw(v) for v in all_golds01], dtype=float)

        def mse(a, b):
            a = np.asarray(a)
            b = np.asarray(b)
            return float(np.mean((a - b) ** 2))

        pearson_norm, _ = pearsonr(all_preds01, all_golds01)
        mse_norm = mse(all_preds01, all_golds01)

        pearson_raw, _ = pearsonr(all_preds_raw, all_golds_raw)
        mse_raw = mse(all_preds_raw, all_golds_raw)

        metrics = {
            "num_pairs_scored": int(all_preds01.size),
            # normalized 0..1
            "pearson_correlation_norm": float(pearson_norm),
            "mse_norm": float(mse_norm),
            # raw scale (dataset native)
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
        }

        preds_path = os.path.join(cfg.out_dir, "predictions.csv")
        metrics_path = os.path.join(cfg.out_dir, "metrics.json")
        icl_path = os.path.join(cfg.out_dir, "icl_examples.json")

        write_csv(preds_path, all_rows)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(icl_path, "w", encoding="utf-8") as f:
            json.dump(icl_examples, f, indent=2)

        print(f"\nSaved predictions to: {preds_path}")
        print(f"Saved metrics to:     {metrics_path}")
        print(f"Saved ICL examples to:{icl_path}")

    # device-aware final barrier, then teardown
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
