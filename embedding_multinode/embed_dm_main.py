# embedding_multinode/embed_dm_main.py
from __future__ import annotations
import argparse
import importlib
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sys
import yaml
import atexit
import torch.multiprocessing as mp
import faulthandler
import signal

# Minimal ZeRO-3 embedder (no monkey patches)
from .big_embeddings_zero3 import DeepSpeedZero3Embeddings


# --------------------------- Config ---------------------------


@dataclass
class RunConfig:
    # Required
    data_module: str  # e.g., "data.stsb:STSBDataModule"
    model: str  # HF model id (informational/logging)
    out_dir: str  # directory to save .npy files
    data_kwargs: Dict[str, Any]

    # Generic
    dtype: str = "bfloat16"
    max_length: int = 4096
    normalize: bool = False
    batch_size: int = 64
    save_labels: bool = False
    splits: Optional[List[str]] = None
    cache_dir: Optional[str] = None

    # Backend (we implement ds_zero3)
    backend: str = "ds_zero3"

    # ZeRO-3 specifics (converter-free)
    model_local_dir: Optional[str] = None  # REQUIRED: local HF snapshot dir
    ds_config_path: Optional[str] = None  # REQUIRED: DS ZeRO-3 json (inference)


# --------------------------- Utils ---------------------------


# put this near the top-level imports (you already have `import os` there)
def _durable_save_npy(path: str, array: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.save(f, array)
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            # fsync may not exist / be needed on some FS
            pass


class _Tee:
    """Write everything to both the original stream and a file (line-buffered)."""

    def __init__(self, stream, filepath: str):
        self._stream = stream
        # buffering=1 -> line buffered; errors='replace' avoids Unicode crashes
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

    def isatty(self):  # tqdm checks this
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


def load_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    for k in ("data_module", "model", "out_dir"):
        if k not in raw:
            raise ValueError(f"Missing required config key: '{k}'")

    raw.setdefault("data_kwargs", {})
    raw.setdefault("dtype", "bfloat16")
    raw.setdefault("max_length", 4096)
    raw.setdefault("normalize", False)
    raw.setdefault("batch_size", 64)
    raw.setdefault("save_labels", False)
    raw.setdefault("splits", None)
    raw.setdefault("cache_dir", None)

    raw.setdefault("backend", "ds_zero3")
    raw.setdefault("model_local_dir", None)
    raw.setdefault(
        "ds_config_path", "embedding_multinode/ds_zero3_inference.json"
    )

    return RunConfig(**raw)


def init_dist() -> tuple[int, int, int]:
    # Populate common envs from MPI/SLURM if present
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
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(minutes=60),
            init_method="env://",
        )

    # Device-aware barrier to avoid c10d warnings
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()

    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), local_rank


def load_datamodule(
    path_cls: str, data_kwargs: Dict[str, Any], model_name: str
):
    module_path, cls_name = path_cls.split(":")
    mod = importlib.import_module(module_path)
    DMCls = getattr(mod, cls_name)
    dm = DMCls(model_name_or_path=model_name, **(data_kwargs or {}))
    dm.setup()
    return dm


def make_loader(dm, split: str, batch_size: int) -> Optional[DataLoader]:
    if split not in dm.dataset:
        return None
    collate = None if getattr(dm, "tokenize_inputs", True) else dm._collate_raw
    # pin_mem = torch.cuda.is_available()
    pin_mem = False
    num_workers = 0  # load data sequentially on main process
    return DataLoader(
        dm.dataset[split],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        collate_fn=collate,
        persistent_workers=(num_workers > 0),
    )


def iter_with_progress(loader, desc: str, rank: int):
    total = len(loader) if hasattr(loader, "__len__") else None
    # If RANK0_LOG is set, send tqdm output there; else stderr.
    log_stream = None
    if rank == 0 and (fp := os.environ.get("RANK0_LOG")):
        # Open a separate handle just for tqdm (line-buffered)
        log_stream = open(
            fp, "a", buffering=1, encoding="utf-8", errors="replace"
        )
    stream = log_stream if log_stream else sys.stderr

    pbar = tqdm(
        total=total,
        desc=desc,
        disable=(rank != 0),
        dynamic_ncols=False,
        leave=True,
        mininterval=0.1,  # update at least every 100ms
        smoothing=0.1,  # faster EMA
        file=stream,
        ascii=True,  # avoid wide/unicode redraws on non-tty pipes
        position=0,
        colour=None,  # avoid ANSI if your launcher strips colors
        ncols=100,
        bar_format=None,
        initial=0,
        unit="it",
        unit_scale=False,
        unit_divisor=1000,
        maxinterval=1.0,  # force periodic refresh under low activity
    )
    try:
        for step, batch in enumerate(loader):
            yield step, batch
            if rank == 0:
                pbar.update(1)
    finally:
        if rank == 0:
            pbar.close()
            if log_stream:
                try:
                    log_stream.flush()
                    log_stream.close()
                except Exception:
                    pass


# --------------------------- Main ---------------------------


def main() -> None:
    try:
        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGBUS)
        faulthandler.register(signal.SIGSEGV)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    # Avoid fork + /dev/shm usage before any DataLoader is constructed
    try:
        mp.set_start_method("spawn", force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    cfg = load_config(args.config)
    rank, world, local_rank = init_dist()

    # If we are rank0 and RANK0_LOG is set, tee *all* prints (incl. HF/tqdm) to the file.
    if rank == 0:
        log_path = os.environ.get("RANK0_LOG")
        if log_path:
            sys.stdout = _Tee(sys.stdout, log_path)
            sys.stderr = _Tee(sys.stderr, log_path)

    # Validate ZeRO-3 path
    if cfg.backend.lower() != "ds_zero3":
        raise ValueError(
            "Only 'ds_zero3' backend is implemented in this minimal overhaul."
        )
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

    dm = load_datamodule(cfg.data_module, cfg.data_kwargs, model_name=cfg.model)

    embedder = None
    try:
        embedder = DeepSpeedZero3Embeddings(
            model_id=cfg.model,
            model_local_dir=cfg.model_local_dir,  # local HF snapshot
            dtype=cfg.dtype,
            max_length=cfg.max_length if cfg.max_length > 0 else 4096,
            normalize=cfg.normalize,
            pad_to_multiple_of=8,
            trust_remote_code=True,
            ds_config_path=cfg.ds_config_path,
            cache_dir=cfg.cache_dir,
        )
    except Exception:
        # propagate error after trying a collective abort
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        raise

    # --- warmup before hitting real data ---
    if rank == 0:
        print("[warmup] starting tiny forward...", flush=True)
    try:
        warmup_s = embedder.warmup(seq_len=min(256, cfg.max_length))
        if rank == 0:
            print(f"[warmup] first forward took {warmup_s:.2f}s", flush=True)
    except Exception as e:
        if rank == 0:
            print(f"[warmup] skipped due to: {e}", flush=True)

    encode_mode = "tokenized" if getattr(dm, "tokenize_inputs", True) else "raw"

    splits = cfg.splits or ["train", "validation", "test"]
    splits = [s for s in splits if s in dm.dataset]
    _rank0_print(f"Processing splits: {splits}")
    os.makedirs(cfg.out_dir, exist_ok=True)

    for split in splits:
        loader = make_loader(dm, split, cfg.batch_size)
        if loader is None:
            _rank0_print(f"Skip missing split: {split}")
            continue

        embs_chunks: List[np.ndarray] = []
        labels_chunks: List[np.ndarray] = []

        with torch.inference_mode():
            for step, batch in iter_with_progress(
                loader, desc=f"[{split}]", rank=rank
            ):
                if encode_mode == "tokenized":
                    # keep it simple: no extra .pin_memory() here (we set pin_memory=False in the loader)
                    token_batch = {
                        k: v
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    token_batch = {
                        k: v.to("cuda", non_blocking=True)
                        for k, v in token_batch.items()
                    }
                    vecs = embedder.encode_tokenized(token_batch).cpu().numpy()
                else:
                    texts = batch["text"]
                    vecs = np.asarray(
                        embedder.embed_documents(list(texts)), dtype=np.float32
                    )

                embs_chunks.append(vecs)

                if cfg.save_labels and "labels" in batch:
                    labels = batch["labels"]
                    arr = (
                        labels.detach().cpu().numpy()
                        if isinstance(labels, torch.Tensor)
                        else np.asarray(labels)
                    )
                    labels_chunks.append(arr.astype(np.float32))

                if rank == 0 and (
                    (step + 1) % max(1, (1000 // cfg.batch_size)) == 0
                ):
                    _rank0_print(
                        f"{split}: processed {(step + 1) * cfg.batch_size} examples"
                    )

        # all ranks finished computing this split before any file I/O
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()

        if rank == 0:
            emb_mat = (
                np.concatenate(embs_chunks, axis=0)
                if embs_chunks
                else np.zeros((0, 0), dtype=np.float32)
            )
            emb_path = os.path.join(cfg.out_dir, f"{split}_embeds.npy")
            _durable_save_npy(emb_path, emb_mat.astype(np.float32))
            _rank0_print(f"Wrote {emb_mat.shape} -> {emb_path}")

            if cfg.save_labels and labels_chunks:
                y = np.concatenate(labels_chunks, axis=0)
                y_path = os.path.join(cfg.out_dir, f"labels_{split}.npy")
                _durable_save_npy(y_path, y.astype(np.float32))
                _rank0_print(f"Wrote labels {y.shape} -> {y_path}")

        # ensure rank0 completes writes before anyone tears down NCCL
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
    # -------- end replacement block --------

    # Clean exit (keep as you had)
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
