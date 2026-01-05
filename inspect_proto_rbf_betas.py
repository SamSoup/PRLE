#!/usr/bin/env python
"""
Inspect learned ProtoRBF betas vs anchor labels.

Assumes you have a saved ProtoRBF model file (e.g. proto_rbf_model.npz)
containing at least:

  - proto_indices: (P,) int, indices into TRAIN embeddings
  - betas:         (P,) float, learned prototype scalars
  - tau:           float (optional)

We then:
  - Rebuild EmbeddingDataModule from the config
  - Get train_labels
  - Compute anchor_labels = train_labels[proto_indices]
  - Compare betas vs anchor_labels (correlations, MAE, etc.)

Usage:

python inspect_proto_rbf_betas.py \
  --config configs/stsb/PRLE/llama_3.1_8B_Instr.yaml \
  --model_path /path/to/proto_rbf_model.npz \
  --top_k 10
"""

import os
import argparse
import numpy as np
import torch

from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr, kendalltau

from data.factory import get_datamodule
from data.embedding import EmbeddingDataModule
from model.encoders.factory import get_encoder


# ---------------------------------------------------------------------
# Embedding DataModule reconstruction
# ---------------------------------------------------------------------


def build_embedding_dm(cfg):
    """
    Same logic as your training script:
      - build raw DataModule from cfg.data + cfg.model
      - wrap with EmbeddingDataModule
      - load / compute cached embeddings + labels
    """
    dataset_name = cfg.data.name
    max_seq_length = cfg.data.max_seq_length
    embedding_cache_dir = cfg.data.embedding_cache_dir

    combine_fields = bool(getattr(cfg.data, "combine_fields", False))
    combine_separator_token = getattr(
        cfg.data, "combine_separator_token", "[SEP]"
    )

    encoder_type = str(cfg.model.encoder_type).lower()
    encoder_name = cfg.model.encoder_name

    train_batch_size = cfg.train.train_batch_size
    eval_batch_size = cfg.train.get("eval_batch_size", train_batch_size)

    tokenize_inputs = encoder_type != "sentence"

    # Raw DataModule
    raw_dm = get_datamodule(
        dataset_name=dataset_name,
        model_name=encoder_name,
        max_seq_length=max_seq_length,
        batch_size=train_batch_size,
        tokenize_inputs=tokenize_inputs,
        combine_fields=combine_fields,
        combine_separator_token=combine_separator_token,
    )

    # Encoder (frozen)
    encoder_kwargs = {}
    if encoder_type == "sentence":
        encoder_kwargs.update(
            cache_dir=getattr(cfg.model, "cache_dir", None),
            normalize_embeddings=getattr(
                cfg.model, "normalize_embeddings", False
            ),
            batch_size=train_batch_size,
        )
    elif encoder_type == "mean":
        encoder_kwargs.update(
            cache_dir=getattr(cfg.model, "cache_dir", None),
            normalize=getattr(cfg.model, "normalize", False),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = None
    # encoder = get_encoder(
    #     encoder_type=encoder_type,
    #     model_name=encoder_name,
    #     **encoder_kwargs,
    # )
    # if isinstance(encoder, torch.nn.Module):
    #     encoder.to(device)
    #     encoder.eval()
    #     for p in encoder.parameters():
    #         p.requires_grad = False

    # EmbeddingDataModule
    embed_dm = EmbeddingDataModule(
        raw_dm=raw_dm,
        encoder=encoder,
        embedding_cache_dir=embedding_cache_dir,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        device=device,
    )
    embed_dm.setup("fit")
    embed_dm.setup("test")
    return embed_dm


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect ProtoRBF betas vs anchor labels."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (OmegaConf style).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved ProtoRBF model npz (with proto_indices, betas).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="How many prototypes to print, sorted by |beta - anchor_label|.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # Build EmbeddingDataModule and get TRAIN labels
    embed_dm = build_embedding_dm(cfg)

    train_labels = (
        embed_dm.train_labels.detach().cpu().numpy().astype(np.float32)
    )  # (N_train,)

    print("[info] train_labels shape:", train_labels.shape)

    # Load ProtoRBF npz
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    print(f"[info] loading ProtoRBF model from {args.model_path}")
    data = np.load(args.model_path, allow_pickle=True)

    if "proto_indices" not in data or "betas" not in data:
        raise KeyError(
            "Expected 'proto_indices' and 'betas' in npz file, but not found."
        )

    proto_indices = data["proto_indices"].astype(int)  # (P,)
    betas = data["betas"].astype(np.float32)  # (P,)
    tau = data["tau"].item() if "tau" in data else None

    print("[info] P (num prototypes):", proto_indices.shape[0])
    if tau is not None:
        print(f"[info] tau: {tau:.4f}")

    # Compute anchor labels
    anchor_labels = train_labels[proto_indices]  # (P,)

    # Sanity
    assert anchor_labels.shape == betas.shape

    # Metrics
    diffs = betas - anchor_labels
    abs_diffs = np.abs(diffs)

    mae = float(abs_diffs.mean())
    max_abs = float(abs_diffs.max())
    pearson_r, _ = pearsonr(anchor_labels, betas)
    spearman_r, _ = spearmanr(anchor_labels, betas)
    kendall_tau, _ = kendalltau(anchor_labels, betas)

    print("\n=== ProtoRBF: betas vs anchor labels ===")
    print(f"MAE(|beta - anchor_label|): {mae:.4f}")
    print(f"Max |beta - anchor_label| : {max_abs:.4f}")
    print(f"Pearson(anchor, beta)     : {pearson_r:.4f}")
    print(f"Spearman(anchor, beta)    : {spearman_r:.4f}")
    print(f"Kendall τ(anchor, beta)   : {kendall_tau:.4f}")

    # Print top-k prototypes by absolute difference
    P = proto_indices.shape[0]
    k = min(args.top_k, P)

    order = np.argsort(-abs_diffs)  # descending by |diff|
    print(f"\nTop-{k} prototypes by |beta - anchor_label|:")
    print("idx\ttrain_idx\tanchor_y\tbeta\t\tdiff")
    for rank in range(k):
        j = int(order[rank])
        train_idx = int(proto_indices[j])
        ay = float(anchor_labels[j])
        b = float(betas[j])
        d = float(diffs[j])
        print(f"{rank:02d}\t{train_idx:7d}\t{ay:8.4f}\t{b:8.4f}\t{d:+8.4f}")

    print("\n[done] ProtoRBF beta inspection complete.")


if __name__ == "__main__":
    main()
