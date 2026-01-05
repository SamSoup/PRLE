#!/usr/bin/env python
"""
Oracle MoE check in value space.

Goal:
    Test whether "perfect" cluster-based local experts actually help.

Procedure (all in value space):

  1. Load config YAML (same style as train.py).
  2. Rebuild EmbeddingDataModule to get retrieval embeddings + labels.
  3. Load stage-1 KernelMetricLearner checkpoint and project to value space.
  4. Run k-medoids with P clusters on TRAIN value embeddings.
  5. Use medoids to assign TRAIN+VAL points to clusters (for training experts),
     and TEST points to clusters (for evaluation).
  6. Train either:
        - global SVR + one SVR per cluster, or
        - global LinearRegression + one LinearRegression per cluster.
  7. Evaluate on TEST set:
        - global regressor (single)
        - oracle MoE: pick cluster by nearest medoid, use that cluster's expert
     Compare MSE, RMSE, Pearson, Spearman, Kendall's tau.

If the oracle MoE does NOT beat the global regressor, local experts probably
don't add much signal in this dataset+encoder+kernel combo.

Example:

python oracle_moe_value_space.py \
  --config config/stsb/PRLE_v2/llama_3.1_8B_Instr.yaml \
  --num_prototypes 25 \
  --regressor_type svr \
  --svr_kernel rbf \
  --svr_C 10.0 \
  --svr_gamma scale \
  --svr_epsilon 0.05
  
python oracle_moe_value_space.py \
  --config config/stsb/PRLE_v2/llama_3.1_8B_Instr.yaml \
  --num_prototypes 25 \
  --regressor_type linear
"""

import os
import argparse
import numpy as np
import torch

from omegaconf import OmegaConf

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn_extra.cluster import KMedoids

from data.factory import get_datamodule
from data.embedding import EmbeddingDataModule
from model.encoders.factory import get_encoder
from model.kernel import KernelMetricLearner


# ============================================================================
# Argparse
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oracle MoE check in value space (k-medoids + local experts)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (OmegaConf style).",
    )
    parser.add_argument(
        "--num_prototypes",
        type=int,
        default=25,
        help="Number of clusters / experts (P) for k-medoids (default: 25).",
    )
    parser.add_argument(
        "--regressor_type",
        type=str,
        default="svr",
        choices=["svr", "linear"],
        help="Type of regressor for global + local experts (default: svr).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for k-medoids and numpy (default: 42).",
    )
    # SVR hyperparameters (used only if regressor_type == 'svr')
    parser.add_argument(
        "--svr_kernel",
        type=str,
        default="rbf",
        choices=["linear", "rbf", "poly", "sigmoid"],
        help="SVR kernel (default: rbf).",
    )
    parser.add_argument(
        "--svr_C",
        type=float,
        default=10.0,
        help="SVR C parameter (default: 10.0).",
    )
    parser.add_argument(
        "--svr_gamma",
        type=str,
        default="scale",
        help='SVR gamma parameter ("scale", "auto", or float as string).',
    )
    parser.add_argument(
        "--svr_epsilon",
        type=float,
        default=0.05,
        help="SVR epsilon parameter (default: 0.05).",
    )
    return parser.parse_args()


# ============================================================================
# Shared helpers (same conventions as analyze script)
# ============================================================================


def build_embedding_dm(cfg):
    """
    Rebuild raw DataModule + EmbeddingDataModule using the same conventions
    as train.py, so we reuse cached embeddings + labels.
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

    # Same logic as train.py: sentence encoders = raw text (no tokenization)
    tokenize_inputs = encoder_type != "sentence"

    # 1. Raw DataModule
    raw_dm = get_datamodule(
        dataset_name=dataset_name,
        model_name=encoder_name,
        max_seq_length=max_seq_length,
        batch_size=train_batch_size,
        tokenize_inputs=tokenize_inputs,
        combine_fields=combine_fields,
        combine_separator_token=combine_separator_token,
    )

    # 2. Encoder (frozen)
    encoder_kwargs = {}
    if encoder_type == "sentence":
        encoder_kwargs.update(
            cache_dir=getattr(cfg.model, "cache_dir", None),
            normalize_embeddings=getattr(
                cfg.model, "normalize_embeddings", False
            ),
            batch_size=train_batch_size,
        )
    if encoder_type == "mean":
        encoder_kwargs.update(
            cache_dir=getattr(cfg.model, "cache_dir", None),
            normalize=getattr(cfg.model, "normalize", False),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = get_encoder(
        encoder_type=encoder_type,
        model_name=encoder_name,
        **encoder_kwargs,
    )
    if isinstance(encoder, torch.nn.Module):
        encoder.to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

    # 3. EmbeddingDataModule
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
    """
    Returns retrieval-space embeddings + labels as numpy arrays:
      (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """

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
    """
    Reconstruct kernel checkpoint path the same way as train.py:

      {checkpoint_dir}/{dataset}_{encoder_type}_{encoder_name}_{projection_type}.ckpt
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


def load_kernel_projection_head(ckpt_path):
    """
    Load KernelMetricLearner and return its projection_head (CPU, eval mode).
    """
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
    """
    X_retrieval: np.ndarray (N, H)
    Returns: np.ndarray (N, Dv) in value space (L2-normalized).
    """
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


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norm


# ============================================================================
# Regressors + evaluation
# ============================================================================


def build_svr_pipeline(kernel, C, gamma_str, epsilon):
    """
    Build a Pipeline: StandardScaler -> SVR.
    """
    if gamma_str in ("scale", "auto"):
        gamma = gamma_str
    else:
        gamma = float(gamma_str)

    svr = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", svr),
        ]
    )
    return pipe


def eval_regressor(estimator, X_eval, y_eval, label=""):
    """
    Evaluate any regressor or Pipeline on X_eval, y_eval.
    Returns metrics dict.
    """
    y_pred = estimator.predict(X_eval)

    y_true = np.asarray(y_eval, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    kendall_tau, _ = kendalltau(y_true, y_pred)

    print(
        f"[{label}] MSE={mse:.6f}  RMSE={rmse:.6f}  "
        f"Pearson={pearson_r:.4f}  Spearman={spearman_r:.4f}  Kendall={kendall_tau:.4f}"
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "pearson": float(pearson_r),
        "spearman": float(spearman_r),
        "kendall": float(kendall_tau),
    }


# ============================================================================
# k-medoids clustering + assignments
# ============================================================================


def run_kmedoids_on_train(X_train, num_prototypes, random_state):
    """
    X_train: (N_train, D) value-space embeddings.
    Returns:
      medoid_indices: (P,) indices into X_train
      medoids_norm  : (P,D) normalized medoid vectors
      labels_train  : (N_train,) cluster labels 0..P-1
    """
    X_norm = _normalize_rows(X_train)
    print(
        f"[kmedoids] running k-medoids on TRAIN, N={X_norm.shape[0]}, "
        f"D={X_norm.shape[1]}, P={num_prototypes}"
    )
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
        medoid_idx = np.asarray(km.medoid_indices_, dtype=int)
    elif getattr(km, "medoid_indices", None) is not None:
        medoid_idx = np.asarray(km.medoid_indices, dtype=int)
    else:
        raise RuntimeError("KMedoids did not expose medoid indices.")

    labels_train = np.asarray(km.labels_, dtype=int)
    medoids_norm = X_norm[medoid_idx]

    print(f"[kmedoids] medoid indices shape: {medoid_idx.shape}")
    print("[kmedoids] cluster sizes (TRAIN):", np.bincount(labels_train))

    return medoid_idx, medoids_norm, labels_train


def assign_to_medoids(X, medoids_norm):
    """
    Assign each row in X to nearest medoid by Euclidean distance
    (assuming both X and medoids_norm are already normalized).
    """
    X_norm = _normalize_rows(X)
    # (N, P) pairwise distances
    diffs = X_norm[:, None, :] - medoids_norm[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return np.argmin(dists, axis=1)


# ============================================================================
# MAIN
# ============================================================================


def main():
    args = parse_args()
    np.random.seed(args.random_state)

    cfg = OmegaConf.load(args.config)

    # 1) Build EmbeddingDataModule & get retrieval embeddings + labels
    embed_dm = build_embedding_dm(cfg)
    (X_train_retr, y_train), (X_val_retr, y_val), (X_test_retr, y_test) = (
        get_retrieval_embeddings_and_labels(embed_dm)
    )

    print("[data] retrieval space shapes:")
    print(f"  train: {X_train_retr.shape}, labels: {y_train.shape}")
    print(f"  val  : {X_val_retr.shape}, labels: {y_val.shape}")
    print(f"  test : {X_test_retr.shape}, labels: {y_test.shape}")

    # 2) Project to value space
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

    # Merge train+val for training regressors (no CV)
    X_train_full_val = np.concatenate([X_train_val, X_val_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    # 3) k-medoids on TRAIN only
    P = int(args.num_prototypes)
    medoid_idx, medoids_norm, train_labels = run_kmedoids_on_train(
        X_train_val,
        num_prototypes=P,
        random_state=args.random_state,
    )

    # 4) Assign TRAIN+VAL and TEST to nearest medoid
    train_full_labels = assign_to_medoids(X_train_full_val, medoids_norm)
    test_labels = assign_to_medoids(X_test_val, medoids_norm)

    print("[assign] cluster sizes (TRAIN+VAL):", np.bincount(train_full_labels))
    print("[assign] cluster sizes (TEST)     :", np.bincount(test_labels))

    # 5) Build global regressor + cluster-local experts
    if args.regressor_type == "svr":
        print(
            "\n===== Global SVR baseline (value space, TRAIN+VAL → TEST) ====="
        )
        global_reg = build_svr_pipeline(
            kernel=args.svr_kernel,
            C=args.svr_C,
            gamma_str=args.svr_gamma,
            epsilon=args.svr_epsilon,
        )
    else:
        print(
            "\n===== Global LinearRegression baseline (value space, TRAIN+VAL → TEST) ====="
        )
        global_reg = LinearRegression()

    global_reg.fit(X_train_full_val, y_train_full)
    metrics_global = eval_regressor(
        global_reg,
        X_test_val,
        y_test,
        label=f"global_{args.regressor_type}/TEST",
    )

    # Train local experts
    print(
        f"\n===== Training local experts (regressor_type={args.regressor_type}, P={P}) ====="
    )
    experts = []
    cluster_sizes = []

    for p in range(P):
        idx = np.where(train_full_labels == p)[0]
        cluster_sizes.append(len(idx))
        if len(idx) == 0:
            print(f"[warn] cluster {p} has no TRAIN+VAL points, skipping.")
            experts.append(None)
            continue

        X_p = X_train_full_val[idx]
        y_p = y_train_full[idx]

        if args.regressor_type == "svr":
            reg_p = build_svr_pipeline(
                kernel=args.svr_kernel,
                C=args.svr_C,
                gamma_str=args.svr_gamma,
                epsilon=args.svr_epsilon,
            )
        else:
            reg_p = LinearRegression()

        reg_p.fit(X_p, y_p)
        experts.append(reg_p)

    print("[experts] cluster sizes (TRAIN+VAL):", cluster_sizes)

    # 6) Oracle MoE prediction on TEST
    print("\n===== Oracle MoE evaluation on TEST =====")
    y_pred_moe = np.zeros_like(y_test, dtype=float)

    for p in range(P):
        reg_p = experts[p]
        if reg_p is None:
            # Fallback: use global regressor if a cluster somehow had no points
            reg_p = global_reg

        mask = test_labels == p
        if not np.any(mask):
            continue

        X_test_p = X_test_val[mask]
        y_pred_p = reg_p.predict(X_test_p)
        y_pred_moe[mask] = y_pred_p

    # Evaluate MoE predictions
    mse = mean_squared_error(y_test, y_pred_moe)
    rmse = float(np.sqrt(mse))
    pearson_r, _ = pearsonr(y_test, y_pred_moe)
    spearman_r, _ = spearmanr(y_test, y_pred_moe)
    kendall_tau, _ = kendalltau(y_test, y_pred_moe)

    metrics_moe = {
        "mse": mse,
        "rmse": rmse,
        "pearson": float(pearson_r),
        "spearman": float(spearman_r),
        "kendall": float(kendall_tau),
    }

    print(
        f"[oracle_moe/TEST] MSE={mse:.6f}  RMSE={rmse:.6f}  "
        f"Pearson={pearson_r:.4f}  Spearman={spearman_r:.4f}  Kendall={kendall_tau:.4f}"
    )

    # 7) Summary
    print("\n=== Oracle MoE vs global baseline (value space, TEST) ===")
    print(f"Regressor type: {args.regressor_type}")
    print("Global:", metrics_global)
    print("MoE   :", metrics_moe)
    print(
        "\nIf MoE > Global (e.g., higher Pearson / lower MSE), local experts are "
        "adding value; otherwise, the space may already be well-modeled by a "
        "single smooth regressor."
    )


if __name__ == "__main__":
    main()
