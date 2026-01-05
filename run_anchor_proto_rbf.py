#!/usr/bin/env python
"""
Anchor-only ProtoRBF (no beta learning; betas = anchor labels).

Steps:
  1) Build EmbeddingDataModule from YAML config (same as train.py).
  2) Load retrieval-space embeddings (train/val/test) + labels.
  3) Load kernel projection head (KernelMetricLearner) and project into value space.
  4) In value space, run K-medoids (or other init) on TRAIN to get prototypes.
  5) Anchor-only betas: beta_p = label of prototype's anchor example.
  6) Tune tau on validation MSE over a small grid.
  7) Evaluate on test.
  8) Optionally save proto_indices, anchor_betas, tau, and metrics.

Usage:

python run_anchor_proto_rbf.py \
  --config configs/stsb/PRLE/llama_3.1_8B_Instr.yaml \
  --output_dir /work/06782/ysu707/ls6/PRLE/anchor_proto_rbf \
  --proto_init kmedoids \
  --tau_values 0.1 0.3 1.0 3.0 10.0
"""

import os
import argparse
import json
import numpy as np
import torch

from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans

from data.factory import get_datamodule
from data.embedding import EmbeddingDataModule
from model.encoders.factory import get_encoder
from model.kernel import KernelMetricLearner


# ---------------------------------------------------------------------
# Embedding + kernel helpers
# ---------------------------------------------------------------------


def build_embedding_dm(cfg):
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


def get_retrieval_embeddings_and_labels(embed_dm):
    def to_np(t: torch.Tensor):
        return t.detach().cpu().numpy()

    X_train = to_np(embed_dm.train_embeddings)
    y_train = to_np(embed_dm.train_labels).astype(np.float32).reshape(-1)

    X_val = to_np(embed_dm.val_embeddings)
    y_val = to_np(embed_dm.val_labels).astype(np.float32).reshape(-1)

    X_test = to_np(embed_dm.test_embeddings)
    y_test = to_np(embed_dm.test_labels).astype(np.float32).reshape(-1)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_kernel_ckpt_path_from_cfg(cfg):
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


def load_kernel_projection_head(ckpt_path):
    print(f"[kernel] loading KernelMetricLearner from {ckpt_path}")
    kernel_model = KernelMetricLearner.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=False,
    )
    kernel_model.eval()
    for p in kernel_model.parameters():
        p.requires_grad = False
    return kernel_model.projection_head


def project_to_value_space(projection_head, X_retrieval, batch_size=4096):
    projection_head.eval()
    all_out = []
    with torch.no_grad():
        for start in range(0, X_retrieval.shape[0], batch_size):
            end = min(start + batch_size, X_retrieval.shape[0])
            chunk = torch.from_numpy(X_retrieval[start:end]).float()
            val_chunk = projection_head(chunk)
            all_out.append(val_chunk.cpu().numpy())
    Z = np.concatenate(all_out, axis=0)
    # L2 normalize
    norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    Z = Z / norm
    return Z


# ---------------------------------------------------------------------
# Prototype helpers (numpy versions)
# ---------------------------------------------------------------------


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norm


def _farthest_first_indices(X_norm: np.ndarray, k: int, random_state: int):
    N = X_norm.shape[0]
    if k >= N:
        return np.arange(N, dtype=int)

    rng = np.random.RandomState(random_state)
    idx0 = int(rng.randint(0, N))
    chosen = [idx0]

    dmin = 1.0 - (X_norm @ X_norm[idx0])
    for _ in range(1, k):
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, 1.0 - (X_norm @ X_norm[j]))
    return np.array(chosen, dtype=int)


def _nearest_point_indices(
    X_norm: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    dmat = np.linalg.norm(
        X_norm[:, None, :] - centers[None, :, :],
        axis=2,
    )  # (N,k)

    N, K = dmat.shape
    used = set()
    out = []
    for j in range(K):
        order = np.argsort(dmat[:, j])
        pick = None
        for ii in order:
            if int(ii) not in used:
                pick = int(ii)
                used.add(pick)
                break
        if pick is None:
            pick = int(order[0])
        out.append(pick)
    return np.array(out, dtype=int)


def compute_prototypes(
    X_train: np.ndarray,
    num_prototypes: int,
    method: str,
    random_state: int,
):
    print(
        f"[protos] method={method}, P={num_prototypes}, N_train={X_train.shape[0]}"
    )
    X_norm = _normalize_rows(X_train)

    if method == "kmedoids":
        km = KMedoids(
            n_clusters=num_prototypes,
            metric="euclidean",
            method="alternate",
            init="k-medoids++",
            max_iter=300,
            random_state=random_state,
        )
        km.fit(X_norm)
        if getattr(km, "medoid_indices_", None) is not None:
            idx = np.asarray(km.medoid_indices_, dtype=int)
        elif getattr(km, "medoid_indices", None) is not None:
            idx = np.asarray(km.medoid_indices, dtype=int)
        else:
            centers = getattr(km, "cluster_centers_", None)
            if centers is not None:
                centers = np.asarray(centers, dtype=float)
                centers = _normalize_rows(centers)
                idx = _nearest_point_indices(X_norm, centers)
            else:
                idx = _farthest_first_indices(
                    X_norm, num_prototypes, random_state
                )
        uniq = list(dict.fromkeys(idx.tolist()))
        if len(uniq) < num_prototypes:
            extra = _farthest_first_indices(
                X_norm, min(num_prototypes, X_norm.shape[0]), random_state
            ).tolist()
            pool = [i for i in extra if i not in set(uniq)]
            uniq = uniq + pool[: (num_prototypes - len(uniq))]
        proto_indices = np.array(uniq[:num_prototypes], dtype=int)

    elif method == "kmeanspp":
        km = KMeans(
            n_clusters=num_prototypes,
            init="k-means++",
            n_init=num_prototypes,
            random_state=random_state,
        )
        km.fit(X_norm)
        centers = np.asarray(km.cluster_centers_, dtype=float)
        centers = _normalize_rows(centers)
        proto_indices = _nearest_point_indices(X_norm, centers)

    elif method == "farthest_first":
        proto_indices = _farthest_first_indices(
            X_norm, num_prototypes, random_state
        )

    else:
        raise ValueError(f"Unknown proto init method: {method}")

    print(f"[protos] selected indices (train-only): {proto_indices.shape}")
    return proto_indices


# ---------------------------------------------------------------------
# Anchor-only RBF predictor
# ---------------------------------------------------------------------


def rbf_softmax_weights(Z: np.ndarray, C: np.ndarray, tau: float):
    """
    Z: (N,D) value-space points
    C: (P,D) prototype centers in value space
    tau: scalar

    Returns:
        W: (N,P) softmax over -tau * ||Z - C||^2 per row
    """
    # Squared Euclidean distances
    # ||z - c||^2 = ||z||^2 + ||c||^2 - 2 z·c
    z_norm2 = np.sum(Z * Z, axis=1, keepdims=True)  # (N,1)
    c_norm2 = np.sum(C * C, axis=1, keepdims=True).T  # (1,P)
    d_sq = z_norm2 + c_norm2 - 2.0 * (Z @ C.T)  # (N,P)
    d_sq = np.maximum(d_sq, 0.0)

    logits = -tau * d_sq  # (N,P)
    # softmax
    logits -= logits.max(axis=1, keepdims=True)  # stabilize
    exp_logits = np.exp(logits)
    W = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
    return W


def predict_anchor_rbf(
    Z: np.ndarray, C: np.ndarray, anchor_betas: np.ndarray, tau: float
):
    """
    Z: (N,D), C: (P,D), anchor_betas: (P,)
    Returns y_hat: (N,)
    """
    W = rbf_softmax_weights(Z, C, tau)  # (N,P)
    y_hat = W @ anchor_betas  # (N,)
    return y_hat


def eval_metrics(y_true, y_pred, label=""):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    pear, _ = pearsonr(y_true, y_pred)
    spear, _ = spearmanr(y_true, y_pred)
    kend, _ = kendalltau(y_true, y_pred)

    print(
        f"[{label}] MSE={mse:.6f}  RMSE={rmse:.6f}  "
        f"Pearson={pear:.4f}  Spearman={spear:.4f}  Kendall={kend:.4f}"
    )
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "pearson": float(pear),
        "spearman": float(spear),
        "kendall": float(kend),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Anchor-only ProtoRBF (betas = anchor labels, tune tau)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (OmegaConf style).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results and model npz.",
    )
    parser.add_argument(
        "--proto_init",
        type=str,
        default="kmedoids",
        choices=["kmedoids", "kmeanspp", "farthest_first"],
        help="Prototype init method in value space (default: kmedoids).",
    )
    parser.add_argument(
        "--tau_values",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 1.0, 3.0, 10.0],
        help="List of tau values to try; pick best on validation MSE.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for prototype selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = OmegaConf.load(args.config)

    np.random.seed(args.random_state)

    # 1) Build embeddings
    embed_dm = build_embedding_dm(cfg)
    (X_train_retr, y_train), (X_val_retr, y_val), (X_test_retr, y_test) = (
        get_retrieval_embeddings_and_labels(embed_dm)
    )

    print("[data] retrieval space shapes:")
    print(f"  train: {X_train_retr.shape}, labels: {y_train.shape}")
    print(f"  val  : {X_val_retr.shape}, labels: {y_val.shape}")
    print(f"  test : {X_test_retr.shape}, labels: {y_test.shape}")

    # 2) Build value-space embeddings using kernel
    kernel_ckpt_path = build_kernel_ckpt_path_from_cfg(cfg)
    print(f"[kernel] expecting checkpoint at: {kernel_ckpt_path}")
    if not os.path.exists(kernel_ckpt_path):
        raise FileNotFoundError(
            f"Kernel checkpoint not found: {kernel_ckpt_path}"
        )

    proj_head = load_kernel_projection_head(kernel_ckpt_path)

    X_train_val = project_to_value_space(proj_head, X_train_retr)
    X_val_val = project_to_value_space(proj_head, X_val_retr)
    X_test_val = project_to_value_space(proj_head, X_test_retr)

    print("[data] value space shapes:")
    print(f"  train: {X_train_val.shape}")
    print(f"  val  : {X_val_val.shape}")
    print(f"  test : {X_test_val.shape}")

    num_prototypes = int(cfg.model.num_prototypes)
    print(f"[config] num_prototypes: {num_prototypes}")

    # 3) Prototypes on TRAIN (value space)
    proto_idx_train = compute_prototypes(
        X_train_val,
        num_prototypes=num_prototypes,
        method=args.proto_init,
        random_state=args.random_state,
    )  # indices into TRAIN

    C_val = X_train_val[proto_idx_train]  # (P,D)
    anchor_betas = y_train[proto_idx_train].astype(np.float32)  # (P,)

    print("[anchors] C_val shape:", C_val.shape)
    print("[anchors] anchor_betas shape:", anchor_betas.shape)

    # 4) Tune tau on validation
    best_tau = None
    best_val_mse = float("inf")
    val_results_by_tau = {}

    print("\n=== Tuning tau on validation ===")
    for tau in args.tau_values:
        y_val_pred = predict_anchor_rbf(X_val_val, C_val, anchor_betas, tau)
        metrics = eval_metrics(y_val, y_val_pred, label=f"val/tau={tau:.4f}")
        val_results_by_tau[tau] = metrics
        if metrics["mse"] < best_val_mse:
            best_val_mse = metrics["mse"]
            best_tau = tau

    print(f"\n[best] tau={best_tau:.4f} with val MSE={best_val_mse:.6f}")

    # 5) Evaluate on test with best tau
    print("\n=== Test metrics with best tau ===")
    y_test_pred = predict_anchor_rbf(X_test_val, C_val, anchor_betas, best_tau)
    test_metrics = eval_metrics(y_test, y_test_pred, label="test/best_tau")

    # 6) Save model + results
    model_npz_path = os.path.join(args.output_dir, "prototypes_and_beta.npz")
    np.savez(
        model_npz_path,
        proto_indices=proto_idx_train,
        betas=anchor_betas,
        tau=best_tau,
        proto_init=args.proto_init,
        num_prototypes=num_prototypes,
    )
    print(f"[save] anchor ProtoRBF model saved to {model_npz_path}")

    results_json_path = os.path.join(args.output_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config_path": args.config,
                "proto_init": args.proto_init,
                "num_prototypes": num_prototypes,
                "tau_values": list(args.tau_values),
                "best_tau": best_tau,
                "val_results_by_tau": val_results_by_tau,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )
    print(f"[save] results JSON saved to {results_json_path}")

    print("\n[done] Anchor-only ProtoRBF run complete.")


if __name__ == "__main__":
    main()
