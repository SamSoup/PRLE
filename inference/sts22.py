# pip install datasets requests scipy numpy --quiet
#
# Usage example:
#   python sts_eval_llm_mteb_sts22_crosslingual.py \
#       --api_key YOUR_KEY_HERE \
#       --model_name Meta-Llama-3.1-8B-Instruct \
#       --out_dir ./results_mteb_sts22 \
#       --limit 200 \
#       --icl_n 5
#
# Notes:
#   - Loads Samsoup/sts22-crosslingual-sts (default subset).
#   - Pairs may be multilingual.
#   - Labels are in [1,4].
#   - We now report metrics on BOTH raw 1..4 and normalized 0..1 via (x-1)/3.

import time
import argparse
import os
import json
import re
import requests
import numpy as np
import random
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm.auto import tqdm

SYSTEM_MSG = (
    "You are an expert annotator for semantic textual similarity (STS) on multilingual sentence pairs.\n"
    "You will see sentence pairs that may be in DIFFERENT languages; judge how similar in MEANING they are.\n\n"
    "Scoring instructions:\n"
    "- Output a real-valued similarity score from 1 to 4, inclusive.\n"
    "- 4 = same meaning / paraphrases; 1 = completely unrelated.\n\n"
    "CRITICAL FORMAT RULE:\n"
    "Return ONLY the numeric score, with up to three digits after the decimal (e.g., 3, 3.1, 3.14, or 3.142).\n"
    "Do NOT include any words, punctuation, units, labels, or explanation.\n"
)


def build_user_message(test_s1, test_s2, icl_examples):
    parts = []
    if icl_examples:
        parts.append(
            "Here are some examples of how to score similarity (pairs may be cross-lingual):\n"
        )
        for i, ex in enumerate(icl_examples, start=1):
            parts.append(
                f"Example {i}:\n"
                f"Sentence A: {ex['sentence1']}\n"
                f"Sentence B: {ex['sentence2']}\n"
                f"Similarity (1..4): {ex['score_1_4_up_to_3']}\n"
            )
        parts.append("\nNow score the new pair.\n")
    parts.append(
        "Sentence A: " + test_s1.strip() + "\n"
        "Sentence B: " + test_s2.strip() + "\n\n"
        "Answer with ONLY the similarity score in the range 1..4 (inclusive), "
        "using up to three digits after the decimal:"
    )
    return "\n".join(parts)


def query_similarity_llm(
    api_base_url,
    sentence_a,
    sentence_b,
    api_key,
    model_name,
    icl_examples,
    timeout=60,
    max_retries=2,
    backoff_seconds=60,
):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    user_msg = build_user_message(sentence_a, sentence_b, icl_examples)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 10,
        "temperature": 0.0,
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                f"{api_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except Exception as e:
            # Network or other request error
            last_error = e
            if attempt == max_retries:
                raise RuntimeError(
                    f"LLM request failed after {max_retries} attempts: {e}"
                )
            print(
                f"Request error (attempt {attempt}/{max_retries}): {e}. "
                f"Sleeping {backoff_seconds} seconds before retry."
            )
            time.sleep(backoff_seconds)
            continue

        # If successful, break out of retry loop
        if resp.status_code == 200:
            break

        # Retry on rate limit or server errors
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            if attempt == max_retries:
                raise RuntimeError(
                    f"LLM request failed after {max_retries} attempts: "
                    f"{resp.status_code} {resp.text}"
                )
            print(
                f"LLM request failed with status {resp.status_code} "
                f"(attempt {attempt}/{max_retries}). "
                f"Sleeping {backoff_seconds} seconds before retry."
            )
            time.sleep(backoff_seconds)
            continue
        else:
            # For other HTTP errors (e.g. 400 / 401), don't retry
            raise RuntimeError(
                f"LLM request failed: {resp.status_code} {resp.text}"
            )

    # At this point, resp.status_code == 200
    data = resp.json()
    raw_answer = None
    try:
        raw_answer = data["choices"][0]["message"]["content"]
    except (KeyError, TypeError):
        raw_answer = data["choices"][0].get("text")
    if raw_answer is None:
        raise RuntimeError(
            f"Could not find completion text in response:\n{data}"
        )

    m = re.search(r"\b\d+(?:\.\d+)?\b", raw_answer.strip())
    if not m:
        raise ValueError(
            f"Could not parse numeric similarity from: {raw_answer}"
        )
    score = float(m.group(0))
    score = max(1.0, min(4.0, score))
    score = float(f"{score:.3f}")
    return score, raw_answer.strip(), data


def mse(pred, gold):
    pred = np.asarray(pred, dtype=float)
    gold = np.asarray(gold, dtype=float)
    return np.mean((pred - gold) ** 2)


def sample_icl_examples(train_split, n):
    if n is None or n <= 0 or train_split is None or len(train_split) == 0:
        return []
    idxs = random.sample(range(len(train_split)), k=min(n, len(train_split)))
    examples = []
    for idx in idxs:
        row = train_split[idx]
        s1 = row["sentence1"]
        s2 = row["sentence2"]
        score_raw = float(row["score"])  # 1..4
        score_txt = f"{score_raw:.3f}"
        examples.append(
            {"sentence1": s1, "sentence2": s2, "score_1_4_up_to_3": score_txt}
        )
    return examples


def run_eval(api_base_url, api_key, model_name, out_dir, limit, icl_n, seed):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    ds = load_dataset("Samsoup/sts22-crosslingual-sts")
    train_split = ds.get("train", None)
    test_split = ds["test"]

    icl_examples = sample_icl_examples(train_split, icl_n)

    gold_scores_raw = []
    pred_scores_raw = []
    gold_scores_norm = []
    pred_scores_norm = []
    rows_for_dump = []

    for i, row in tqdm(enumerate(test_split), total=len(test_split)):
        if limit is not None and i >= limit:
            break
        s1 = row["sentence1"]
        s2 = row["sentence2"]
        gold_raw = float(row["score"])  # 1..4
        gold_norm = (gold_raw - 1.0) / 3.0  # → [0,1]

        try:
            pred_raw, raw_answer, _ = query_similarity_llm(
                api_base_url,
                s1,
                s2,
                api_key=api_key,
                model_name=model_name,
                icl_examples=icl_examples,
            )
        except Exception as e:
            print(f"[{i}] ERROR querying model: {e}")
            continue

        pred_norm = (pred_raw - 1.0) / 3.0

        gold_scores_raw.append(gold_raw)
        pred_scores_raw.append(pred_raw)
        gold_scores_norm.append(gold_norm)
        pred_scores_norm.append(pred_norm)

        rows_for_dump.append(
            {
                "idx": i,
                "sentence1": s1,
                "sentence2": s2,
                "gold_similarity_1_4": float(f"{gold_raw:.3f}"),
                "llm_similarity_1_4": float(f"{pred_raw:.3f}"),
                "gold_similarity_0_1": float(f"{gold_norm:.6f}"),
                "llm_similarity_0_1": float(f"{pred_norm:.6f}"),
                "raw_model_reply": raw_answer,
            }
        )

        if i % 100 == 0:
            print(f"[{i}] Gold 1..4: {gold_raw:.3f} | LLM 1..4: {pred_raw:.3f}")

    gold_scores_raw = np.array(gold_scores_raw, dtype=float)
    pred_scores_raw = np.array(pred_scores_raw, dtype=float)
    gold_scores_norm = np.array(gold_scores_norm, dtype=float)
    pred_scores_norm = np.array(pred_scores_norm, dtype=float)

    # raw metrics
    mse_raw = mse(pred_scores_raw, gold_scores_raw)
    rmse_raw = float(np.sqrt(mse_raw))
    pearson_raw, _ = pearsonr(pred_scores_raw, gold_scores_raw)
    spearman_raw, _ = spearmanr(pred_scores_raw, gold_scores_raw)
    kendall_raw, _ = kendalltau(pred_scores_raw, gold_scores_raw)

    # normalized metrics
    mse_norm = mse(pred_scores_norm, gold_scores_norm)
    rmse_norm = float(np.sqrt(mse_norm))
    pearson_norm, _ = pearsonr(pred_scores_norm, gold_scores_norm)
    spearman_norm, _ = spearmanr(pred_scores_norm, gold_scores_norm)
    kendall_norm, _ = kendalltau(pred_scores_norm, gold_scores_norm)

    metrics = {
        "num_pairs_scored": int(len(pred_scores_raw)),
        "model_name": model_name,
        "icl_n": int(len(icl_examples)),
        "label_range": "1..4 (also reported normalized to 0..1 via (x-1)/3)",
        "raw": {
            "mse": float(mse_raw),
            "rmse": float(rmse_raw),
            "pearson_correlation": float(pearson_raw),
            "spearman_correlation": float(spearman_raw),
            "kendall_correlation": float(kendall_raw),
        },
        "normalized_0_1": {
            "mse": float(mse_norm),
            "rmse": float(rmse_norm),
            "pearson_correlation": float(pearson_norm),
            "spearman_correlation": float(spearman_norm),
            "kendall_correlation": float(kendall_norm),
        },
    }

    print("=======================================")
    print(f"Pairs evaluated: {metrics['num_pairs_scored']}")
    print("--- RAW 1..4 scale ---")
    print(f"Pearson: {metrics['raw']['pearson_correlation']:.4f}")
    print(f"Spearman: {metrics['raw']['spearman_correlation']:.4f}")
    print(f"Kendall τ: {metrics['raw']['kendall_correlation']:.4f}")
    print(f"MSE:  {metrics['raw']['mse']:.6f}")
    print(f"RMSE: {metrics['raw']['rmse']:.6f}")
    print("--- Normalized 0..1 scale ---")
    print(f"Pearson: {metrics['normalized_0_1']['pearson_correlation']:.4f}")
    print(f"Spearman: {metrics['normalized_0_1']['spearman_correlation']:.4f}")
    print(f"Kendall τ: {metrics['normalized_0_1']['kendall_correlation']:.4f}")
    print(f"MSE:  {metrics['normalized_0_1']['mse']:.6f}")
    print(f"RMSE: {metrics['normalized_0_1']['rmse']:.6f}")
    print(f"In-context examples used: {metrics['icl_n']}")

    preds_path = os.path.join(out_dir, "predictions.csv")
    metrics_path = os.path.join(out_dir, "metrics.json")
    icl_path = os.path.join(out_dir, "icl_examples.json")

    with open(preds_path, "w", encoding="utf-8") as f:
        f.write(
            "idx,sentence1,sentence2,gold_similarity_1_4,llm_similarity_1_4,"
            "gold_similarity_0_1,llm_similarity_0_1,raw_model_reply\n"
        )
        for r in rows_for_dump:

            def esc(x):
                return '"' + str(x).replace('"', '""') + '"'

            f.write(
                ",".join(
                    [
                        str(r["idx"]),
                        esc(r["sentence1"]),
                        esc(r["sentence2"]),
                        f"{r['gold_similarity_1_4']:.3f}",
                        f"{r['llm_similarity_1_4']:.3f}",
                        f"{r['gold_similarity_0_1']:.6f}",
                        f"{r['llm_similarity_0_1']:.6f}",
                        esc(r["raw_model_reply"]),
                    ]
                )
                + "\n"
            )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(icl_path, "w", encoding="utf-8") as f:
        json.dump(icl_examples, f, indent=2)

    print(f"\nSaved predictions to: {preds_path}")
    print(f"Saved metrics to:     {metrics_path}")
    print(f"Saved ICL examples to:{icl_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate LLM similarity on Samsoup/sts22-crosslingual-sts (1..4 labels; multilingual pairs)."
    )
    p.add_argument("--api_base_url", type=str, required=True)
    p.add_argument("--api_key", type=str, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--icl_n", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        args.api_base_url,
        args.api_key,
        args.model_name,
        args.out_dir,
        args.limit,
        args.icl_n,
        args.seed,
    )
