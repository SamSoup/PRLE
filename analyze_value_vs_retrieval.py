#!/usr/bin/env python
"""
Analyze retrieval vs value space:

- Loads config YAML (same style as train.py).
- Uses EmbeddingDataModule to load retrieval embeddings + labels.
- Loads stage-1 KernelMetricLearner and projects into value space.
- Optionally:
    (1) Runs SVR (with StandardScaler pipeline) in both spaces.
    (2) Runs LinearRegression in both spaces.
    (3) Computes correlation between |Δy| and distances before/after.
    (4) Runs k-medoids / k-means++ / farthest-first on TRAIN embeddings
        (in both spaces) to get prototype indices.
        Runs t-SNE and saves 2D plots with train/val/test + prototypes.

Example:

python -m analyze_value_vs_retrieval \
  --config config/stsb/PRLE_v2/llama_3.1_8B_Instr.yaml \
  --proto_init kmedoids \
  --tsne_random_state 42 \
  --tsne_perplexity 30.0 \
  --svr_kernel rbf \
  --svr_C 10.0 \
  --svr_gamma scale \
  --svr_epsilon 0.05 \
  --fig_output_dir /work/06782/ysu707/ls6/PRLE/explore_and_debug \
  --do_svr 1 \
  --do_linreg 1 \
  --do_tsne 1
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans

from data.factory import get_datamodule
from data.embedding import EmbeddingDataModule
from model.encoders.factory import get_encoder
from model.kernel import KernelMetricLearner


# ============================================================================
# CONFIG + ARGPARSE
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze retrieval vs value space using config YAML."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (OmegaConf style).",
    )
    parser.add_argument(
        "--tsne_random_state",
        type=int,
        default=42,
        help="Random seed for t-SNE and prototype init (default: 42).",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="Base perplexity for t-SNE (auto-clipped by N).",
    )
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
    parser.add_argument(
        "--fig_output_dir",
        type=str,
        default="/work/06782/ysu707/ls6/PRLE/explore_and_debug",
        help="Directory to save t-SNE plots (default given).",
    )
    parser.add_argument(
        "--proto_init",
        type=str,
        default="kmedoids",
        choices=["kmedoids", "kmeanspp", "farthest_first"],
        help="Prototype init method for analysis (default: kmedoids).",
    )
    # knobs: 1) SVR analysis, 2) Linear regression analysis, 3) t-SNE plots
    parser.add_argument(
        "--do_svr",
        type=int,
        default=1,
        choices=[0, 1],
        help="Run SVR analysis? 1=yes, 0=no (default: 1).",
    )
    parser.add_argument(
        "--do_linreg",
        type=int,
        default=1,
        choices=[0, 1],
        help="Run LinearRegression analysis? 1=yes, 0=no (default: 1).",
    )
    parser.add_argument(
        "--do_tsne",
        type=int,
        default=1,
        choices=[0, 1],
        help="Run t-SNE + prototype plots? 1=yes, 0=no (default: 1).",
    )
    # optional: number of pairs for distance–label correlation
    parser.add_argument(
        "--corr_num_pairs",
        type=int,
        default=50000,
        help="Number of random train pairs for distance–label correlation (default: 50k).",
    )
    return parser.parse_args()


# ============================================================================
# EmbeddingDataModule reconstruction
# ============================================================================


def build_embedding_dm(cfg):
    """
    Rebuild raw DataModule + EmbeddingDataModule using the same conventions
    as train.py, so we reuse cached embeddings + labels.
    """
    # Dataset / encoder config
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


# ============================================================================
# Kernel → value space
# ============================================================================


def build_kernel_ckpt_path_from_cfg(cfg):
    """
    Reconstruct kernel checkpoint path the same way as train.py:

      {checkpoint_dir}/{dataset}_{encoder_type}_{encoder_name}_{projection_type}.ckpt
    using cfg.kernel.checkpoint_dir.
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


# ============================================================================
# Regression pipelines (SVR + LinearRegression)
# ============================================================================


def build_svr_pipeline(kernel, C, gamma_str, epsilon):
    """
    Build a Pipeline: StandardScaler -> SVR.

    gamma_str: "scale", "auto", or float string.
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
        "pearson": pearson_r,
        "spearman": spearman_r,
        "kendall": kendall_tau,
    }


# ============================================================================
# Prototypes: k-medoids / k-means++ / farthest-first (numpy versions)
# ============================================================================


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norm


def _farthest_first_indices(X_norm: np.ndarray, k: int, random_state: int):
    """
    X_norm: (N, D) L2-normalized rows.
    Returns: indices of farthest-first centers (size k).
    """
    N = X_norm.shape[0]
    if k >= N:
        return np.arange(N, dtype=int)

    rng = np.random.RandomState(random_state)
    idx0 = int(rng.randint(0, N))
    chosen = [idx0]

    # cosine distance = 1 - cos_sim
    dmin = 1.0 - (X_norm @ X_norm[idx0])
    for _ in range(1, k):
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, 1.0 - (X_norm @ X_norm[j]))
    return np.array(chosen, dtype=int)


def _nearest_point_indices(
    X_norm: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """
    For each center, pick the nearest unique point index in X_norm.
    X_norm: (N,D)
    centers: (k,D)
    Returns: (k,) indices.
    """
    # Euclidean distances
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
    """
    X_train: (N,H) TRAIN embeddings (retrieval or value).
    method: "kmedoids" | "kmeanspp" | "farthest_first"
    Returns:
        proto_indices: (P,) indices into X_train
    """
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
            # Fallback: use nearest points to centers if present
            centers = getattr(km, "cluster_centers_", None)
            if centers is not None:
                centers = np.asarray(centers, dtype=float)
                centers = _normalize_rows(centers)
                idx = _nearest_point_indices(X_norm, centers)
            else:
                idx = _farthest_first_indices(
                    X_norm, num_prototypes, random_state
                )
        # ensure uniqueness
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


# ============================================================================
# t-SNE plotting
# ============================================================================


def tsne_embed(X, n_components=2, perplexity=30.0, random_state=42):
    n_samples = X.shape[0]
    effective_perp = min(perplexity, max(5.0, (n_samples - 1) / 3.0))
    print(
        f"[tsne] N={n_samples}, D={X.shape[1]}, perplexity={effective_perp:.1f}"
    )
    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perp,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X)


def plot_tsne_with_prototypes(
    Z_2d,
    y,
    split_ids,
    proto_indices,
    title,
    fig_path=None,
):
    """
    Z_2d: (N,2) t-SNE coords for all points
    y: (N,) labels
    split_ids: (N,) array of {"train", "val", "test"} strings
    proto_indices: indices (relative to ALL points) to mark as prototypes
    """
    plt.figure(figsize=(8, 6))
    y = np.asarray(y)

    # Normalize labels to [0,1] for coloring
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min + 1e-8)

    split_ids = np.asarray(split_ids)

    for split, marker, alpha in [
        ("train", "o", 0.6),
        ("val", "^", 0.8),
        ("test", "s", 0.8),
    ]:
        mask = split_ids == split
        if not np.any(mask):
            continue
        plt.scatter(
            Z_2d[mask, 0],
            Z_2d[mask, 1],
            c=y_norm[mask],
            cmap="viridis",
            marker=marker,
            s=10,
            alpha=alpha,
            label=split,
        )

    if proto_indices is not None and len(proto_indices) > 0:
        proto_coords = Z_2d[proto_indices]
        plt.scatter(
            proto_coords[:, 0],
            proto_coords[:, 1],
            c="red",
            marker="*",
            s=120,
            edgecolors="black",
            linewidths=1.0,
            label="prototypes",
        )

    plt.colorbar(label="label (normalized)")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()

    if fig_path is not None:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=200)
        print(f"[plot] saved to {fig_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# Distance–label correlation
# ============================================================================


def compute_label_distance_correlation(
    X,
    y,
    num_pairs=50000,
    random_state=42,
    label="",
):
    """
    Sample random pairs from TRAIN and compute correlation between
    |Δy| and Euclidean distances on L2-normalized embeddings.
    """
    X_norm = _normalize_rows(X)
    y = np.asarray(y, dtype=float).reshape(-1)
    N = X_norm.shape[0]

    num_pairs = min(num_pairs, N * (N - 1) // 2)
    rng = np.random.RandomState(random_state)
    idx1 = rng.randint(0, N, size=num_pairs)
    idx2 = rng.randint(0, N, size=num_pairs)

    # allow duplicates / self-pairs; they just add tiny noise
    diff_vecs = X_norm[idx1] - X_norm[idx2]
    dists = np.linalg.norm(diff_vecs, axis=1)
    delta_y = np.abs(y[idx1] - y[idx2])

    pearson_r, _ = pearsonr(delta_y, dists)
    spearman_r, _ = spearmanr(delta_y, dists)

    print(
        f"[dist-corr/{label}] Pearson(|Δy|, dist)={pearson_r:.4f}  "
        f"Spearman={spearman_r:.4f}"
    )

    return {
        "pearson": pearson_r,
        "spearman": spearman_r,
    }


# ============================================================================
# MAIN
# ============================================================================


def main():
    args = parse_args()
    os.makedirs(args.fig_output_dir, exist_ok=True)

    cfg = OmegaConf.load(args.config)

    np.random.seed(args.tsne_random_state)

    do_svr = bool(args.do_svr)
    do_linreg = bool(args.do_linreg)
    do_tsne = bool(args.do_tsne)

    # 1) Build EmbeddingDataModule & load retrieval embeddings/labels
    embed_dm = build_embedding_dm(cfg)
    (X_train_retr, y_train), (X_val_retr, y_val), (X_test_retr, y_test) = (
        get_retrieval_embeddings_and_labels(embed_dm)
    )

    print("[data] retrieval space shapes:")
    print(f"  train: {X_train_retr.shape}, labels: {y_train.shape}")
    print(f"  val  : {X_val_retr.shape}, labels: {y_val.shape}")
    print(f"  test : {X_test_retr.shape}, labels: {y_test.shape}")

    # 2) Build value-space embeddings using kernel projection head
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

    # ------------------------------------------------------------------
    # 3) Distance–label correlation (train only) before vs after kernel
    # ------------------------------------------------------------------
    print("\n===== Distance–label correlation on TRAIN =====")
    corr_retr = compute_label_distance_correlation(
        X_train_retr,
        y_train,
        num_pairs=args.corr_num_pairs,
        random_state=args.tsne_random_state,
        label="retrieval",
    )
    corr_val = compute_label_distance_correlation(
        X_train_val,
        y_train,
        num_pairs=args.corr_num_pairs,
        random_state=args.tsne_random_state,
        label="value",
    )
    print("dist-corr retrieval:", corr_retr)
    print("dist-corr value    :", corr_val)

    # ------------------------------------------------------------------
    # 4) Regression baselines (TEST ONLY)
    # ------------------------------------------------------------------
    X_train_full_retr = np.concatenate([X_train_retr, X_val_retr], axis=0)
    X_train_full_val = np.concatenate([X_train_val, X_val_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    if do_svr:
        print("\n===== SVR in retrieval vs value space (TEST only) =====")
        # retrieval
        svr_retr = build_svr_pipeline(
            kernel=args.svr_kernel,
            C=args.svr_C,
            gamma_str=args.svr_gamma,
            epsilon=args.svr_epsilon,
        )
        svr_retr.fit(X_train_full_retr, y_train_full)
        metrics_svr_retr = eval_regressor(
            svr_retr, X_test_retr, y_test, label="SVR/retrieval/TEST"
        )

        # value
        svr_val = build_svr_pipeline(
            kernel=args.svr_kernel,
            C=args.svr_C,
            gamma_str=args.svr_gamma,
            epsilon=args.svr_epsilon,
        )
        svr_val.fit(X_train_full_val, y_train_full)
        metrics_svr_val = eval_regressor(
            svr_val, X_test_val, y_test, label="SVR/value/TEST"
        )

        print("\n=== SVR summary (TEST) ===")
        print("Retrieval TEST:", metrics_svr_retr)
        print("Value     TEST:", metrics_svr_val)

    if do_linreg:
        print(
            "\n===== LinearRegression in retrieval vs value space (TEST only) ====="
        )
        # retrieval (no scaling, matches wrapperbox style)
        lin_retr = LinearRegression()
        lin_retr.fit(X_train_full_retr, y_train_full)
        metrics_lin_retr = eval_regressor(
            lin_retr, X_test_retr, y_test, label="LinReg/retrieval/TEST"
        )

        # value
        lin_val = LinearRegression()
        lin_val.fit(X_train_full_val, y_train_full)
        metrics_lin_val = eval_regressor(
            lin_val, X_test_val, y_test, label="LinReg/value/TEST"
        )

        print("\n=== LinearRegression summary (TEST) ===")
        print("Retrieval TEST:", metrics_lin_retr)
        print("Value     TEST:", metrics_lin_val)

    # ------------------------------------------------------------------
    # 5) t-SNE + prototype plots
    # ------------------------------------------------------------------
    if do_tsne:
        # Prototypes on TRAIN in retrieval vs value space
        proto_idx_retr_train = compute_prototypes(
            X_train_retr,
            num_prototypes=num_prototypes,
            method=args.proto_init,
            random_state=args.tsne_random_state,
        )
        proto_idx_val_train = compute_prototypes(
            X_train_val,
            num_prototypes=num_prototypes,
            method=args.proto_init,
            random_state=args.tsne_random_state,
        )

        # In concatenated arrays, TRAIN is first chunk → indices map directly
        proto_indices_retr_all = proto_idx_retr_train
        proto_indices_val_all = proto_idx_val_train

        # t-SNE in retrieval space
        X_all_retr = np.concatenate(
            [X_train_retr, X_val_retr, X_test_retr], axis=0
        )
        y_all = np.concatenate([y_train, y_val, y_test], axis=0)
        split_ids = (
            ["train"] * len(y_train)
            + ["val"] * len(y_val)
            + ["test"] * len(y_test)
        )
        split_ids = np.array(split_ids)

        Z_retr_2d = tsne_embed(
            X_all_retr,
            n_components=2,
            perplexity=args.tsne_perplexity,
            random_state=args.tsne_random_state,
        )
        plot_tsne_with_prototypes(
            Z_retr_2d,
            y_all,
            split_ids,
            proto_indices_retr_all,
            title=f"t-SNE in retrieval space ({args.proto_init}, P={num_prototypes})",
            fig_path=os.path.join(
                args.fig_output_dir,
                f"{args.proto_init}/tsne_retrieval_space.png",
            ),
        )

        # t-SNE in value space
        X_all_val = np.concatenate([X_train_val, X_val_val, X_test_val], axis=0)
        Z_val_2d = tsne_embed(
            X_all_val,
            n_components=2,
            perplexity=args.tsne_perplexity,
            random_state=args.tsne_random_state,
        )
        plot_tsne_with_prototypes(
            Z_val_2d,
            y_all,
            split_ids,
            proto_indices_val_all,
            title=f"t-SNE in value space ({args.proto_init}, P={num_prototypes})",
            fig_path=os.path.join(
                args.fig_output_dir, f"{args.proto_init}/tsne_value_space.png"
            ),
        )

    print(
        "\n[done] Correlations + regression metrics (and optional t-SNE plots) "
        "computed for retrieval vs value spaces."
    )


if __name__ == "__main__":
    main()
