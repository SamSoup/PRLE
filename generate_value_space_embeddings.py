#!/usr/bin/env python
"""
Project retrieval-space embeddings into value space using a trained kernel.

Pipeline:

1) Assume retrieval embeddings are already cached as:
       {retrieval_cache_dir}/train_embeds.npy
       {retrieval_cache_dir}/validation_embeds.npy
       {retrieval_cache_dir}/test_embeds.npy
   (these are produced by EmbeddingDataModule in train.py)

2) Load the KernelMetricLearner checkpoint (projection head).

3) Project each split into value space (with L2 normalization).

4) Save value-space embeddings under a new directory, e.g.:

       {value_cache_dir}/train_embeds.npy
       {value_cache_dir}/validation_embeds.npy
       {value_cache_dir}/test_embeds.npy

Then you can run your existing wrapper baseline script on BOTH dirs:

    # retrieval space
    python run_wrapperbox.py \
        --emb_dir /scratch/.../stsb/llama_3.1_8B_Instr \
        --dataset_name <HF_DATASET> \
        --output_dir <OUT_RETRIEVAL>

    # value space
    python run_wrapperbox.py \
        --emb_dir /scratch/.../stsb/llama_3.1_8B_Instr_value \
        --dataset_name <HF_DATASET> \
        --output_dir <OUT_VALUE>

Usage:

    python make_value_space_embeddings.py \
        --config configs/stsb/PRLE/llama_3.1_8B_Instr.yaml \
        --value_cache_dir /scratch/.../stsb/llama_3.1_8B_Instr_value \
        --batch_size 4096
"""

import os
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf

from model.kernel import KernelMetricLearner


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project retrieval embeddings into value space using a trained kernel."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (OmegaConf style).",
    )
    parser.add_argument(
        "--retrieval_cache_dir",
        type=str,
        default=None,
        help=(
            "Directory containing retrieval embeddings "
            "(train_embeds.npy, validation_embeds.npy, test_embeds.npy). "
            "If omitted, defaults to cfg.data.embedding_cache_dir."
        ),
    )
    parser.add_argument(
        "--value_cache_dir",
        type=str,
        required=True,
        help=(
            "Directory where value-space embeddings will be written "
            "(train_embeds.npy, validation_embeds.npy, test_embeds.npy)."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size when projecting embeddings through the kernel.",
    )
    return parser.parse_args()


def build_kernel_ckpt_path_from_cfg(cfg) -> str:
    """
    Match the checkpoint naming convention used in train.py:

        {kernel.checkpoint_dir}/{data.name}_{model.encoder_type}_{
            model.encoder_name with '/' -> '-'
        }_{kernel.projection_type or model.projection_type}.ckpt
    """
    dataset_name = cfg.data.name
    encoder_type_str = cfg.model.encoder_type
    encoder_name_str = str(cfg.model.encoder_name).replace("/", "-")

    proj_type = getattr(
        cfg.kernel,
        "projection_type",
        getattr(cfg.model, "projection_type", "resmlp"),
    )
    ckpt_dir = cfg.kernel.checkpoint_dir
    ckpt_name = (
        f"{dataset_name}_{encoder_type_str}_{encoder_name_str}_{proj_type}.ckpt"
    )
    return os.path.join(ckpt_dir, ckpt_name)


def load_projection_head(ckpt_path: str):
    """
    Load KernelMetricLearner from checkpoint and return its projection_head
    on CPU in eval mode.
    """
    print(f"[kernel] loading KernelMetricLearner from: {ckpt_path}")
    kernel_model = KernelMetricLearner.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=False,
    )
    kernel_model.eval()
    for p in kernel_model.parameters():
        p.requires_grad = False
    return kernel_model.projection_head


def project_split(
    proj_head: torch.nn.Module,
    X_retrieval: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Project a whole split from retrieval → value space with L2 normalization.

    Args:
        proj_head: torch module (projection_head)
        X_retrieval: np.ndarray of shape (N, H)
        batch_size: chunk size for memory

    Returns:
        Z_value: np.ndarray of shape (N, Dv)
    """
    proj_head.eval()
    all_out = []

    with torch.no_grad():
        for start in range(0, X_retrieval.shape[0], batch_size):
            end = min(start + batch_size, X_retrieval.shape[0])
            chunk = torch.from_numpy(X_retrieval[start:end]).float()
            val_chunk = proj_head(chunk)  # (B, Dv)
            all_out.append(val_chunk.cpu().numpy())

    Z = np.concatenate(all_out, axis=0)
    # L2 normalize rows (matches PrototypeManager.project_embeddings behavior)
    norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    Z = Z / norm
    return Z.astype(np.float32)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    retrieval_cache_dir = (
        args.retrieval_cache_dir
        if args.retrieval_cache_dir is not None
        else cfg.data.embedding_cache_dir
    )
    value_cache_dir = args.value_cache_dir

    os.makedirs(value_cache_dir, exist_ok=True)

    print(f"[config] using retrieval cache dir: {retrieval_cache_dir}")
    print(f"[config] writing value-space cache dir: {value_cache_dir}")

    # -----------------------------------------------------------------
    # 1) Load retrieval-space embeddings
    # -----------------------------------------------------------------
    def load_split(split_name: str) -> np.ndarray:
        path = os.path.join(retrieval_cache_dir, f"{split_name}_embeds.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Retrieval embeddings not found for split '{split_name}': {path}"
            )
        arr = np.load(path)
        print(f"[load] {split_name}: {arr.shape}")
        return arr

    X_train_retr = load_split("train")
    X_val_retr = load_split("validation")
    X_test_retr = load_split("test")

    # -----------------------------------------------------------------
    # 2) Load kernel projection head
    # -----------------------------------------------------------------
    kernel_ckpt_path = build_kernel_ckpt_path_from_cfg(cfg)
    print(f"[kernel] expecting checkpoint at: {kernel_ckpt_path}")
    if not os.path.exists(kernel_ckpt_path):
        raise FileNotFoundError(
            f"Kernel checkpoint not found: {kernel_ckpt_path}"
        )

    proj_head = load_projection_head(kernel_ckpt_path)

    # -----------------------------------------------------------------
    # 3) Project each split into value space
    # -----------------------------------------------------------------
    print("\n[project] projecting TRAIN → value space")
    X_train_val = project_split(
        proj_head, X_train_retr, batch_size=args.batch_size
    )
    print("[project] TRAIN value space:", X_train_val.shape)

    print("\n[project] projecting VALIDATION → value space")
    X_val_val = project_split(proj_head, X_val_retr, batch_size=args.batch_size)
    print("[project] VAL value space:", X_val_val.shape)

    print("\n[project] projecting TEST → value space")
    X_test_val = project_split(
        proj_head, X_test_retr, batch_size=args.batch_size
    )
    print("[project] TEST value space:", X_test_val.shape)

    # -----------------------------------------------------------------
    # 4) Save value-space embeddings
    # -----------------------------------------------------------------
    def save_split(split_name: str, arr: np.ndarray):
        out_path = os.path.join(value_cache_dir, f"{split_name}_embeds.npy")
        np.save(out_path, arr.astype(np.float32))
        print(f"[save] {split_name}: {arr.shape} → {out_path}")

    save_split("train", X_train_val)
    save_split("validation", X_val_val)
    save_split("test", X_test_val)

    print("\n[done] Value-space embeddings written.")
    print("       Now you can run run_wrapperbox.py with:")
    print(f"         --emb_dir {value_cache_dir}")
    print("       to evaluate baselines in value space.")


if __name__ == "__main__":
    main()
