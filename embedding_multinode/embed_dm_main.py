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
import yaml

# Backends
from .big_embeddings_zero3 import (
    DeepSpeedZero3Embeddings,
)  # converter-free ZeRO-3 path

# (Optional: keep your previous classes if you want other backends later)
# from .big_embeddings import DeepSpeedShardedEmbeddings
# from .big_embeddings_vllm import VLLMRemoteEmbeddings


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

    # Backend selection (we default to ZeRO-3)
    backend: str = "ds_zero3"

    # ZeRO-3 specifics (converter-free)
    model_local_dir: Optional[str] = None  # REQUIRED: local HF snapshot dir
    ds_config_path: Optional[str] = None  # REQUIRED: DS ZeRO-3 json (inference)


# --------------------------- Utils ---------------------------


def iter_with_progress(loader, desc: str, rank: int):
    """
    Iterate over a DataLoader and render a tqdm progress bar on rank 0 only.
    The bar counts batches (not examples) to stay cheap and robust.
    """
    total = len(loader) if hasattr(loader, "__len__") else None
    pbar = tqdm(
        total=total,
        desc=desc,
        disable=(rank != 0),
        dynamic_ncols=True,
        leave=False,
    )
    try:
        for step, batch in enumerate(loader):
            yield step, batch
            if rank == 0:
                pbar.update(1)
    finally:
        if rank == 0:
            pbar.close()


def _log0(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def load_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    for k in ("data_module", "model", "out_dir"):
        if k not in raw:
            raise ValueError(f"Missing required config key: '{k}'")

    # Defaults
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
        pg_kwargs = dict(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(minutes=60),
            init_method="env://",
        )
        # PyTorch >= 2.1 supports device_id in init_process_group
        if torch.cuda.is_available():
            try:
                pg_kwargs["device_id"] = local_rank
            except TypeError:
                # older torch: no device_id arg
                pass
        dist.init_process_group(**pg_kwargs)

    # Device-aware barrier to avoid NCCL warnings
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
    pin_mem = torch.cuda.is_available()
    # good rule: 2–4 workers per GPU; tune if you have CPU headroom
    num_workers = 4
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


# --------------------------- Main ---------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    # before main() or early in main()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")  # PyTorch 2.x
    torch.backends.cudnn.benchmark = True

    cfg = load_config(args.config)
    rank, world, local_rank = init_dist()

    # Early validation for ZeRO-3 path (no server, no manifest)
    if cfg.backend.lower() == "ds_zero3":
        if not cfg.model_local_dir or not os.path.isdir(cfg.model_local_dir):
            raise FileNotFoundError(
                f"model_local_dir not found: {cfg.model_local_dir}"
            )
        if not cfg.ds_config_path or not os.path.isfile(cfg.ds_config_path):
            raise FileNotFoundError(
                f"ds_config_path not found: {cfg.ds_config_path}"
            )

    _log0(rank, f"[world={world} local_rank={local_rank}]")
    _log0(rank, f"Config: {cfg}")

    # Data module
    dm = load_datamodule(cfg.data_module, cfg.data_kwargs, model_name=cfg.model)

    # Backend selection (we implement ds_zero3 here)
    backend = cfg.backend.lower()
    if backend == "ds_zero3":
        embedder = DeepSpeedZero3Embeddings(
            model_local_dir=cfg.model_local_dir,  # local HF snapshot
            dtype=cfg.dtype,
            max_length=cfg.max_length if cfg.max_length > 0 else None,
            normalize=cfg.normalize,
            pad_to_multiple_of=8,
            trust_remote_code=True,
            ds_config_path=cfg.ds_config_path,
            cache_dir=cfg.cache_dir,
        )
        encode_mode = (
            "tokenized" if getattr(dm, "tokenize_inputs", True) else "raw"
        )

    else:
        raise ValueError(
            f"Unknown backend: {cfg.backend}. Expected 'ds_zero3'."
        )

    # Splits & output dir
    splits = cfg.splits or ["train", "validation", "test"]
    splits = [s for s in splits if s in dm.dataset]
    _log0(rank, f"Processing splits: {splits}")
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Iterate splits
    for split in splits:
        loader = make_loader(dm, split, cfg.batch_size)
        if loader is None:
            _log0(rank, f"Skip missing split: {split}")
            continue

        embs_chunks: List[np.ndarray] = []
        labels_chunks: List[np.ndarray] = []

        with torch.inference_mode():  # cheaper than no_grad
            for step, batch in iter_with_progress(
                loader, desc=f"[{split}]", rank=rank
            ):
                if encode_mode == "tokenized":
                    token_batch = {
                        k: v
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    # if you want to overlap H2D copies:
                    token_batch = {
                        k: v.pin_memory().to("cuda", non_blocking=True)
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

            # occasional textual heartbeat (kept)
            if rank == 0 and (
                (step + 1) % max(1, (1000 // cfg.batch_size)) == 0
            ):
                _log0(
                    rank,
                    f"{split}: processed {(step + 1) * cfg.batch_size} examples",
                )

        # Save (rank 0)
        if rank == 0:
            emb_mat = (
                np.concatenate(embs_chunks, axis=0)
                if embs_chunks
                else np.zeros((0, 0), dtype=np.float32)
            )
            emb_path = os.path.join(cfg.out_dir, f"embeddings_{split}.npy")
            np.save(emb_path, emb_mat.astype(np.float32))
            _log0(rank, f"Wrote {emb_mat.shape} -> {emb_path}")

            if cfg.save_labels and labels_chunks:
                y = np.concatenate(labels_chunks, axis=0)
                y_path = os.path.join(cfg.out_dir, f"labels_{split}.npy")
                np.save(y_path, y.astype(np.float32))
                _log0(rank, f"Wrote labels {y.shape} -> {y_path}")

    # Clean exit
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
