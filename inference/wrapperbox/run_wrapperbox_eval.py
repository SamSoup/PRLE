#!/usr/bin/env python

"""
Usage:

python run_wrapperbox.py \
  --emb_dir /scratch/06782/ysu707/PRLE/wmt_en_ru/llama_33_70b_instr \
  --dataset_name Samsoup/Samsoup-WMT2020-ru-en \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/wmt_en_ru/llama_33_70b_instr/wrapperbox

"""

import os
import json
import argparse

import numpy as np
import pandas as pd

from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# -------------------------------------------------------------------
# Evaluation helper (extended)
# -------------------------------------------------------------------
def eval_on_test(estimator, X_test, y_test):
    """
    Works for plain estimators or Pipeline(scaler -> regressor).

    Returns:
        mse          : float
        rmse         : float
        pearson_r    : float
        spearman_r   : float
        kenda ll_tau  : float
        y_pred       : np.ndarray
    """
    y_pred = estimator.predict(X_test)

    # Ensure 1-D arrays for scipy
    y_true = np.asarray(y_test, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    kendall_tau, _ = kendalltau(y_true, y_pred)

    return (
        round(mse, 3),
        round(rmse, 3),
        round(pearson_r, 3),
        round(spearman_r, 3),
        round(kendall_tau, 3),
        y_pred,
    )


# -------------------------------------------------------------------
# CV helper
# -------------------------------------------------------------------
def tune_with_validation_cv(
    X_train,
    y_train,
    X_val,
    y_val,
    build_base_estimator_fn,
    param_grid,
    scale=False,
    scoring="neg_mean_squared_error",
    verbose=2,
):
    """
    Use sklearn's GridSearchCV with a manual train/val split.

    - If scale=True: we build
          Pipeline([("scaler", StandardScaler()),
                    ("regressor", base_estimator)])
      and param_grid keys must be 'regressor__<paramname>'.

    - If scale=False: we pass the base estimator directly, and param_grid
      should use plain parameter names.
    """
    # combine train + val for GridSearch with PredefinedSplit
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)

    split_index = np.concatenate(
        [
            -1 * np.ones(len(X_train), dtype=int),  # training fold
            0 * np.ones(len(X_val), dtype=int),  # validation fold
        ]
    )
    predefined = PredefinedSplit(test_fold=split_index)

    base_estimator = build_base_estimator_fn()
    if scale:
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", base_estimator),
            ]
        )
    else:
        estimator = base_estimator

    gscv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=predefined,
        scoring=scoring,
        verbose=verbose,
        n_jobs=-1,
    )
    gscv.fit(X_all, y_all)

    best_estimator = gscv.best_estimator_
    best_params = gscv.best_params_
    best_val_mse = -gscv.best_score_  # since scoring = neg_mse

    print("===== GridSearch Summary =====")
    print("Best params:", best_params)
    print("Best validation MSE:", best_val_mse)
    print("==============================")

    return best_estimator, best_params, best_val_mse


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run classical regression baselines on precomputed embeddings "
            "for a HF regression dataset (train/validation/test)."
        )
    )
    parser.add_argument(
        "--emb_dir",
        type=str,
        required=True,
        help="Base directory containing train_embeds.npy, validation_embeds.npy, test_embeds.npy.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help='HuggingFace dataset name, e.g. "Samsoup/Samsoup-WMT2020-ru-en".',
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
        "--verbose_cv",
        type=int,
        default=2,
        help="Verbosity level for GridSearchCV (default: 2).",
    )
    args = parser.parse_args()

    emb_dir = args.emb_dir
    dataset_name = args.dataset_name
    out_dir = args.output_dir
    score_col = args.score_column
    verbose_cv = args.verbose_cv

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
    # Load dataset labels from HF
    # ------------------------------
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    y_train = np.array(ds["train"][score_col], dtype=float)
    y_val = np.array(ds["validation"][score_col], dtype=float)
    y_test = np.array(ds["test"][score_col], dtype=float)

    print("Shapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
    print("  X_test: ", X_test.shape, "y_test: ", y_test.shape)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    # Merge train + val for final training of some models
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    print(
        "X_train_full:", X_train_full.shape, "y_train_full:", y_train_full.shape
    )

    # Container for results
    results = []

    # ---------------------------------------------------------
    # 1) Linear Regression (no scaling, no hyperparams)
    # ---------------------------------------------------------
    print("\n=== Linear Regression ===")
    linreg = LinearRegression()
    linreg.fit(X_train_full, y_train_full)

    (
        linreg_mse,
        linreg_rmse,
        linreg_pearson,
        linreg_spearman,
        linreg_kendall,
        _,
    ) = eval_on_test(linreg, X_test, y_test)

    print(f"Test MSE:       {linreg_mse:.4f}")
    print(f"Test RMSE:      {linreg_rmse:.4f}")
    print(f"Test Pearson r: {linreg_pearson:.4f}")
    print(f"Test Spearman:  {linreg_spearman:.4f}")
    print(f"Test Kendall τ: {linreg_kendall:.4f}")

    results.append(
        {
            "model": "LinearRegression",
            "val_mse": None,
            "test_mse": float(linreg_mse),
            "test_rmse": float(linreg_rmse),
            "test_pearson": float(linreg_pearson),
            "test_spearman": float(linreg_spearman),
            "test_kendall": float(linreg_kendall),
            "best_params": None,
        }
    )

    # ---------------------------------------------------------
    # 2) SVR (RBF) with scaling + GridSearchCV
    # ---------------------------------------------------------
    print("\n=== SVR (RBF) with scaling + CV ===")

    def build_svr_base():
        return SVR()

    svr_param_grid = {
        "regressor__kernel": ["rbf"],
        "regressor__C": [0.1, 1.0, 10.0, 100.0],
        "regressor__epsilon": [0.05, 0.1, 0.2],
        "regressor__gamma": ["scale", "auto"],
    }

    svr_best_estimator, svr_best_params, svr_best_val_mse = (
        tune_with_validation_cv(
            X_train,
            y_train,
            X_val,
            y_val,
            build_svr_base,
            svr_param_grid,
            scale=True,
            scoring="neg_mean_squared_error",
            verbose=verbose_cv,
        )
    )

    (
        svr_mse,
        svr_rmse,
        svr_pearson,
        svr_spearman,
        svr_kendall,
        _,
    ) = eval_on_test(svr_best_estimator, X_test, y_test)

    print(f"Best Val MSE:   {svr_best_val_mse:.4f}")
    print(f"Test MSE:       {svr_mse:.4f}")
    print(f"Test RMSE:      {svr_rmse:.4f}")
    print(f"Test Pearson r: {svr_pearson:.4f}")
    print(f"Test Spearman:  {svr_spearman:.4f}")
    print(f"Test Kendall τ: {svr_kendall:.4f}")

    results.append(
        {
            "model": "SVR_RBF",
            "val_mse": float(svr_best_val_mse),
            "test_mse": float(svr_mse),
            "test_rmse": float(svr_rmse),
            "test_pearson": float(svr_pearson),
            "test_spearman": float(svr_spearman),
            "test_kendall": float(svr_kendall),
            "best_params": svr_best_params,
        }
    )

    # ---------------------------------------------------------
    # 3) KNN Regressor with scaling + GridSearchCV
    # ---------------------------------------------------------
    print("\n=== KNN Regressor with scaling + CV ===")

    def build_knn_base():
        return KNeighborsRegressor()

    knn_param_grid = {
        "regressor__n_neighbors": [
            3,
            5,
            7,
            9,
            11,
            20,
            50,
            75,
            100,
            200,
            500,
            1000,
        ],
        "regressor__weights": ["uniform", "distance"],
        "regressor__p": [1, 2],  # L1 or L2
        "regressor__metric": ["euclidean", "cosine"],
    }

    knn_best_estimator, knn_best_params, knn_best_val_mse = (
        tune_with_validation_cv(
            X_train,
            y_train,
            X_val,
            y_val,
            build_knn_base,
            knn_param_grid,
            scale=True,
            scoring="neg_mean_squared_error",
            verbose=verbose_cv,
        )
    )

    (
        knn_mse,
        knn_rmse,
        knn_pearson,
        knn_spearman,
        knn_kendall,
        _,
    ) = eval_on_test(knn_best_estimator, X_test, y_test)

    print(f"Best Val MSE:   {knn_best_val_mse:.4f}")
    print(f"Test MSE:       {knn_mse:.4f}")
    print(f"Test RMSE:      {knn_rmse:.4f}")
    print(f"Test Pearson r: {knn_pearson:.4f}")
    print(f"Test Spearman:  {knn_spearman:.4f}")
    print(f"Test Kendall τ: {knn_kendall:.4f}")

    results.append(
        {
            "model": "KNNRegressor",
            "val_mse": float(knn_best_val_mse),
            "test_mse": float(knn_mse),
            "test_rmse": float(knn_rmse),
            "test_pearson": float(knn_pearson),
            "test_spearman": float(knn_spearman),
            "test_kendall": float(knn_kendall),
            "best_params": knn_best_params,
        }
    )

    # ---------------------------------------------------------
    # 4) Decision Tree Regressor (no scaling, small grid)
    # ---------------------------------------------------------
    print("\n=== DecisionTreeRegressor + CV ===")

    def build_dt_base():
        return DecisionTreeRegressor(random_state=RANDOM_STATE)

    dt_param_grid = {
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
    }

    dt_best_estimator, dt_best_params, dt_best_val_mse = (
        tune_with_validation_cv(
            X_train,
            y_train,
            X_val,
            y_val,
            build_dt_base,
            dt_param_grid,
            scale=False,
            scoring="neg_mean_squared_error",
            verbose=verbose_cv,
        )
    )

    (
        dt_mse,
        dt_rmse,
        dt_pearson,
        dt_spearman,
        dt_kendall,
        _,
    ) = eval_on_test(dt_best_estimator, X_test, y_test)

    print(f"Best Val MSE:   {dt_best_val_mse:.4f}")
    print(f"Test MSE:       {dt_mse:.4f}")
    print(f"Test RMSE:      {dt_rmse:.4f}")
    print(f"Test Pearson r: {dt_pearson:.4f}")
    print(f"Test Spearman:  {dt_spearman:.4f}")
    print(f"Test Kendall τ: {dt_kendall:.4f}")

    results.append(
        {
            "model": "DecisionTreeRegressor",
            "val_mse": float(dt_best_val_mse),
            "test_mse": float(dt_mse),
            "test_rmse": float(dt_rmse),
            "test_pearson": float(dt_pearson),
            "test_spearman": float(dt_spearman),
            "test_kendall": float(dt_kendall),
            "best_params": dt_best_params,
        }
    )

    # ---------------------------------------------------------
    # 5) RandomForestRegressor (no scaling, small grid)
    # ---------------------------------------------------------
    print("\n=== RandomForestRegressor + CV ===")

    def build_rf_base():
        return RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    rf_param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 5],
    }

    rf_best_estimator, rf_best_params, rf_best_val_mse = (
        tune_with_validation_cv(
            X_train,
            y_train,
            X_val,
            y_val,
            build_rf_base,
            rf_param_grid,
            scale=False,
            scoring="neg_mean_squared_error",
            verbose=verbose_cv,
        )
    )

    (
        rf_mse,
        rf_rmse,
        rf_pearson,
        rf_spearman,
        rf_kendall,
        _,
    ) = eval_on_test(rf_best_estimator, X_test, y_test)

    print(f"Best Val MSE:   {rf_best_val_mse:.4f}")
    print(f"Test MSE:       {rf_mse:.4f}")
    print(f"Test RMSE:      {rf_rmse:.4f}")
    print(f"Test Pearson r: {rf_pearson:.4f}")
    print(f"Test Spearman:  {rf_spearman:.4f}")
    print(f"Test Kendall τ: {rf_kendall:.4f}")

    results.append(
        {
            "model": "RandomForestRegressor",
            "val_mse": float(rf_best_val_mse),
            "test_mse": float(rf_mse),
            "test_rmse": float(rf_rmse),
            "test_pearson": float(rf_pearson),
            "test_spearman": float(rf_spearman),
            "test_kendall": float(rf_kendall),
            "best_params": rf_best_params,
        }
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------
    df_results = pd.DataFrame(results)

    # Sort: high Pearson first, then low MSE
    df_results = df_results.sort_values(
        by=["test_pearson", "test_mse"],
        ascending=[False, True],
    ).reset_index(drop=True)

    csv_path = os.path.join(out_dir, "results.csv")
    json_path = os.path.join(out_dir, "results.json")

    df_results.to_csv(csv_path, index=False)

    # Convert best_params to plain dicts / strings for JSON
    serializable_results = []
    for row in results:
        row_copy = dict(row)
        if isinstance(row_copy.get("best_params"), dict):
            row_copy["best_params"] = {
                str(k): (
                    v if isinstance(v, (int, float, str, bool)) else str(v)
                )
                for k, v in row_copy["best_params"].items()
            }
        serializable_results.append(row_copy)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "embedding_dir": emb_dir,
                "score_column": score_col,
                "results": serializable_results,
            },
            f,
            indent=2,
        )

    print("\n=== Saved outputs ===")
    print("CSV: ", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
