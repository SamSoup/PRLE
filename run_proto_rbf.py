#!/usr/bin/env python
"""
Prototype RBF regressor over K-medoids prototypes with an anchor prior on betas.

Usage (value space example):

python run_prototype_rbf.py \
  --emb_dir /scratch/06782/ysu707/PRLE/stsb/llama_3.1_8B_Instr/value_space \
  --dataset_name glue/stsb \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/stsb/llama_3.1_8B_Instr/proto_rbf_value_anchor \
  --score_column label \
  --num_prototypes 25 \
  --tau_grid 0.1 0.3 1.0 3.0 10.0 \
  --lambda_anchor 1.0

You can also run on retrieval space by pointing --emb_dir to the retrieval
embedding cache instead.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd

from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn.metrics import mean_squared_error
from sklearn_extra.cluster import KMedoids

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def eval_on_split(W, beta, y_true):
    """
    Evaluate a prototype-RBF model on a given split.

    Args:
        W:      (N, P) RBF weight matrix for this split
        beta:   (P,) prototype scalars
        y_true: (N,) true labels

    Returns:
        mse, rmse, pearson, spearman, kendall
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = W @ beta

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    kendall_tau, _ = kendalltau(y_true, y_pred)

    return (
        float(np.round(mse, 6)),
        float(np.round(rmse, 6)),
        float(np.round(pearson_r, 6)),
        float(np.round(spearman_r, 6)),
        float(np.round(kendall_tau, 6)),
    )


def compute_rbf_weights(Z, centers, tau):
    """
    Compute softmax-normalized RBF weights:

        w_p(z_i) ∝ exp( -tau * ||z_i - c_p||^2 )

    Args:
        Z:       (N, D) data embeddings
        centers: (P, D) prototype centers
        tau:     float > 0

    Returns:
        W: (N, P) weights, rows sum to 1
    """
    # Squared Euclidean distances: (N, P)
    diff = Z[:, None, :] - centers[None, :, :]  # (N, P, D)
    dist2 = np.sum(diff * diff, axis=-1)  # (N, P)

    logits = -tau * dist2
    # Stabilize softmax
    logits = logits - logits.max(axis=1, keepdims=True)
    W = np.exp(logits)
    W_sum = W.sum(axis=1, keepdims=True) + 1e-12
    W = W / W_sum
    return W


def solve_anchor_ridge(W, y, anchor_labels, lambda_anchor):
    """
    Solve for beta in:

        min_beta ||W beta - y||^2 + lambda_anchor ||beta - a||^2

    where a = anchor_labels.

    Closed form:
        (W^T W + λ I) beta = W^T y + λ a
    """
    W = np.asarray(W, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    a = np.asarray(anchor_labels, dtype=float).ravel()
    P = W.shape[1]

    A = W.T @ W + lambda_anchor * np.eye(P, dtype=float)
    b = W.T @ y + lambda_anchor * a

    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback: least-squares solver
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return beta.astype(float)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prototype RBF regressor over K-medoids prototypes with "
            "an anchor prior on betas. Assumes precomputed embeddings "
            "(train/validation/test)."
        )
    )
    parser.add_argument(
        "--emb_dir",
        type=str,
        required=True,
        help=(
            "Directory with train_embeds.npy, validation_embeds.npy, "
            "test_embeds.npy."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help='HF dataset name, e.g. "glue/stsb" or "Samsoup/Samsoup-WMT2020-ru-en".',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write results.csv and results.json.",
    )
    parser.add_argument(
        "--score_column",
        type=str,
        default="score",
        help="Name of the label column in the HF dataset (default: score).",
    )
    parser.add_argument(
        "--num_prototypes",
        type=int,
        default=25,
        help="Number of K-medoids prototypes (default: 25).",
    )
    parser.add_argument(
        "--tau_grid",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 1.0, 3.0, 10.0],
        help="Grid of tau values to try (default: 0.1 0.3 1.0 3.0 10.0).",
    )
    parser.add_argument(
        "--lambda_anchor",
        type=float,
        default=1.0,
        help=(
            "Strength of the prior ||beta - anchor_labels||^2 "
            "(default: 1.0)."
        ),
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for K-medoids (default: 42).",
    )
    args = parser.parse_args()

    emb_dir = args.emb_dir
    dataset_name = args.dataset_name
    out_dir = args.output_dir
    score_col = args.score_column
    P = args.num_prototypes
    tau_grid = args.tau_grid
    lambda_anchor = float(args.lambda_anchor)
    seed = args.random_state

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------
    # Load embeddings
    # ------------------------------
    train_path = os.path.join(emb_dir, "train_embeds.npy")
    val_path = os.path.join(emb_dir, "validation_embeds.npy")
    test_path = os.path.join(emb_dir, "test_embeds.npy")

    print("Loading embeddings from:", emb_dir)
    X_train = np.load(train_path)
    X_val = np.load(val_path)
    X_test = np.load(test_path)

    # ------------------------------
    # Load labels from HF
    # ------------------------------
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    y_train = np.array(ds["train"][score_col], dtype=float)
    y_val = np.array(ds["validation"][score_col], dtype=float)
    y_test = np.array(ds["test"][score_col], dtype=float)

    print("Shapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
    print("  X_test: ", X_test.shape, "y_test:", y_test.shape)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    # ------------------------------
    # K-medoids on TRAIN to get prototypes
    # ------------------------------
    print(f"\n=== K-medoids prototype selection (P={P}) ===")
    km = KMedoids(
        n_clusters=P,
        metric="euclidean",
        method="alternate",
        init="k-medoids++",
        max_iter=300,
        random_state=seed,
    )
    km.fit(X_train)

    if getattr(km, "medoid_indices_", None) is not None:
        medoid_indices = np.asarray(km.medoid_indices_, dtype=int)
    elif getattr(km, "medoid_indices", None) is not None:
        medoid_indices = np.asarray(km.medoid_indices, dtype=int)
    else:
        raise RuntimeError(
            "KMedoids did not expose medoid_indices; please update sklearn_extra."
        )

    print("medoid_indices:", medoid_indices.shape)
    centers = X_train[medoid_indices]  # (P, D)
    anchor_labels = y_train[medoid_indices]  # (P,)

    # ------------------------------
    # Grid search over tau using validation MSE
    # with anchor-regularized betas
    # ------------------------------
    best_tau = None
    best_val_mse = float("inf")
    best_beta = None
    best_metrics_val = None

    print("\n=== Tuning tau on validation (with anchor prior) ===")
    print(f"lambda_anchor = {lambda_anchor}")

    for tau in tau_grid:
        print(f"\n  [tau = {tau}]")

        # Compute RBF weights for train/val
        W_train = compute_rbf_weights(X_train, centers, tau)  # (N_train, P)
        W_val = compute_rbf_weights(X_val, centers, tau)  # (N_val, P)

        # Solve anchor-regularized ridge:
        # (W^T W + λ I) beta = W^T y + λ anchor
        beta = solve_anchor_ridge(
            W_train, y_train, anchor_labels, lambda_anchor
        )

        # Eval on validation
        val_mse, val_rmse, val_pearson, val_spearman, val_kendall = (
            eval_on_split(W_val, beta, y_val)
        )

        print(f"    Val MSE:       {val_mse:.6f}")
        print(f"    Val RMSE:      {val_rmse:.6f}")
        print(f"    Val Pearson r: {val_pearson:.6f}")
        print(f"    Val Spearman:  {val_spearman:.6f}")
        print(f"    Val Kendall τ: {val_kendall:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_tau = tau
            best_beta = beta
            best_metrics_val = {
                "mse": val_mse,
                "rmse": val_rmse,
                "pearson": val_pearson,
                "spearman": val_spearman,
                "kendall": val_kendall,
            }

    print("\n=== Best tau summary ===")
    print(f"Best tau: {best_tau}")
    print(f"Best Val MSE: {best_val_mse:.6f}")
    print("Best Val metrics:", best_metrics_val)

    # ------------------------------
    # Refit beta on train+val with best tau
    # ------------------------------
    print("\n=== Refit on train+val with best tau (anchor prior) ===")
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    W_train_full = compute_rbf_weights(X_train_full, centers, best_tau)
    W_test = compute_rbf_weights(X_test, centers, best_tau)

    beta_full = solve_anchor_ridge(
        W_train_full, y_train_full, anchor_labels, lambda_anchor
    )

    (
        test_mse,
        test_rmse,
        test_pearson,
        test_spearman,
        test_kendall,
    ) = eval_on_split(W_test, beta_full, y_test)

    # Quick sanity print: correlation between beta and anchor labels
    corr_beta_anchor = np.corrcoef(beta_full, anchor_labels)[0, 1]
    print(f"\n[debug] Corr(beta, anchor_labels) = {corr_beta_anchor:.4f}")

    print("\n=== Prototype RBF (anchor-regularized) Test metrics ===")
    print(f"Test MSE:       {test_mse:.6f}")
    print(f"Test RMSE:      {test_rmse:.6f}")
    print(f"Test Pearson r: {test_pearson:.6f}")
    print(f"Test Spearman:  {test_spearman:.6f}")
    print(f"Test Kendall τ: {test_kendall:.6f}")

    # ------------------------------
    # Save results (CSV + JSON) in wrapperbox style
    # ------------------------------
    results = [
        {
            "model": "PrototypeRBF_AnchorRidge",
            "val_mse": float(best_val_mse),
            "test_mse": float(test_mse),
            "test_rmse": float(test_rmse),
            "test_pearson": float(test_pearson),
            "test_spearman": float(test_spearman),
            "test_kendall": float(test_kendall),
            "best_params": {
                "num_prototypes": int(P),
                "tau": float(best_tau),
                "random_state": int(seed),
                "lambda_anchor": float(lambda_anchor),
            },
            "beta_anchor_corr": float(corr_beta_anchor),
        }
    ]

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "results.csv")
    json_path = os.path.join(out_dir, "results.json")

    df_results.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "embedding_dir": emb_dir,
                "score_column": score_col,
                "results": results,
            },
            f,
            indent=2,
        )

    # Save prototypes + beta for interpretability
    protos_path = os.path.join(out_dir, "prototypes_and_beta.npz")
    np.savez(
        protos_path,
        proto_indices=medoid_indices,
        centers=centers,
        betas=beta_full,
        anchor_labels=anchor_labels,
        best_tau=best_tau,
        lambda_anchor=lambda_anchor,
    )

    print("\n=== Saved outputs ===")
    print("CSV:        ", csv_path)
    print("JSON:       ", json_path)
    print("Protos+β:   ", protos_path)


if __name__ == "__main__":
    main()
