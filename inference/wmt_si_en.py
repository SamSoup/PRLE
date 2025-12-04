# pip install datasets requests scipy numpy tqdm --quiet
#
# Usage example:
#   python sts_eval_wmt20_sien.py \
#       --api_base_url http://localhost:8000/v1 \
#       --api_key YOUR_KEY_HERE \
#       --model_name Meta-Llama-3.1-8B-Instruct \
#       --out_dir ./results_wmt20_sien_run1 \
#       --limit 100 \
#       --icl_n 5

import argparse
import os
import json
import time
import re
import requests
import numpy as np
import random
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm.auto import tqdm

HF_DATASET_NAME = "samsoup/Samsoup-WMT2020-si-en"
SCORE_MAX = 100.0

SYSTEM_MSG = (
    "You are an expert human annotator for translation quality / semantic textual similarity (STS).\n"
    "You will see sentence pairs where Sentence A is in Sinhala (source) and Sentence B is in English (translation).\n"
    "You must judge how well the meaning is preserved from Sinhala to English.\n\n"
    "Scoring instructions:\n"
    "- Output a real-valued translation quality / similarity score from 0.00 to 100.00.\n"
    "- 100.00 = perfect meaning preservation (excellent translation / paraphrases).\n"
    "- 50.00  = partially similar meaning but with important differences or missing info.\n"
    "- 0.00   = completely unrelated or very bad translation.\n\n"
    "CRITICAL FORMAT RULE:\n"
    "Return ONLY the numeric score with EXACTLY two digits after the decimal point.\n"
    "Do NOT include any words, punctuation, units, labels, or explanation.\n"
)


def build_user_message(test_s1, test_s2, icl_examples):
    parts = []

    if icl_examples:
        parts.append(
            "Here are some examples of how to score Sinhala (source) to English (translation) quality / similarity:\n"
        )
        for i, ex in enumerate(icl_examples, start=1):
            parts.append(
                f"Example {i}:\n"
                f"Sentence A (Sinhala source): {ex['sentence1']}\n"
                f"Sentence B (English translation): {ex['sentence2']}\n"
                f"Score (0-100): {ex['score_0_100_two_decimals']}\n"
            )
        parts.append("\nNow score the new pair.\n")

    parts.append(
        "Sentence A (Sinhala source): " + test_s1.strip() + "\n"
        "Sentence B (English translation): " + test_s2.strip() + "\n\n"
        "Answer with ONLY the translation quality / similarity score in the range "
        "0.00 to 100.00, using exactly two digits after the decimal point:"
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
    max_retries=3,
    backoff_seconds=60,
):
    """
    Ask the LLM for a 0.00-100.00 similarity score (two decimal places) for the given pair.
    Returns (score_float_raw, raw_answer_str, full_json_dict).

    score_float_raw is clamped to [0, 100] and rounded to 3 decimal places.
    """
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

        if resp.status_code == 200:
            break

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
            raise RuntimeError(
                f"LLM request failed: {resp.status_code} {resp.text}"
            )

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

    m = re.search(r"(-?\d+(?:\.\d+)?)", raw_answer.strip())
    if not m:
        raise ValueError(
            f"Could not parse numeric similarity from: {raw_answer}"
        )

    score_raw = float(m.group(0))
    score_raw = max(0.0, min(100.0, score_raw))
    score_raw = round(score_raw, 3)

    return score_raw, raw_answer.strip(), data


def mse(pred, gold):
    pred = np.asarray(pred, dtype=float)
    gold = np.asarray(gold, dtype=float)
    return np.mean((pred - gold) ** 2)


def sample_icl_examples(train_split, n):
    """
    Sample train examples and format 0-100 gold scores for ICL.
    """
    if n is None or n <= 0:
        return []

    idxs = random.sample(range(len(train_split)), k=min(n, len(train_split)))

    examples = []
    for idx in idxs:
        row = train_split[idx]
        s1 = row["sentence1"]
        s2 = row["sentence2"]
        score_raw = round(float(row["score"]), 3)
        score_txt = f"{score_raw:.2f}"
        examples.append(
            {
                "sentence1": s1,
                "sentence2": s2,
                "score_0_100_two_decimals": score_txt,
            }
        )

    return examples


def run_eval(api_base_url, api_key, model_name, out_dir, limit, icl_n):
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset(HF_DATASET_NAME)
    train_split = ds["train"]
    test_split = ds["test"]

    icl_examples = sample_icl_examples(train_split, icl_n)

    gold_raw_scores = []
    pred_raw_scores = []
    rows_for_dump = []

    for i, row in tqdm(enumerate(test_split), total=len(test_split)):
        if limit is not None and i >= limit:
            break

        s1 = row["sentence1"]
        s2 = row["sentence2"]

        gold_raw = round(float(row["score"]), 3)
        gold_norm = gold_raw / SCORE_MAX

        try:
            pred_raw, raw_answer, raw_json = query_similarity_llm(
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

        pred_norm = pred_raw / SCORE_MAX

        gold_raw_scores.append(gold_raw)
        pred_raw_scores.append(pred_raw)

        rows_for_dump.append(
            {
                "idx": i,
                "sentence1": s1,
                "sentence2": s2,
                "gold_score_raw_0_100": gold_raw,
                "gold_score_norm_0_1": gold_norm,
                "llm_score_raw_0_100": pred_raw,
                "llm_score_norm_0_1": pred_norm,
                "raw_model_reply": raw_answer,
            }
        )

        if i % 100 == 0:
            print(f"[{i}]")
            print("Sentence 1:", s1)
            print("Sentence 2:", s2)
            print("Gold raw:", f"{gold_raw:.3f}")
            print("LLM  raw:", f"{pred_raw:.3f}")
            print("Gold (0-1):", f"{gold_norm:.3f}")
            print("LLM  (0-1):", f"{pred_norm:.3f}")
            print("Raw model reply:", raw_answer)
            print()
            time.sleep(0.25)

    gold_raw_scores = np.array(gold_raw_scores, dtype=float)
    pred_raw_scores = np.array(pred_raw_scores, dtype=float)

    gold_norm_scores = gold_raw_scores / SCORE_MAX
    pred_norm_scores = pred_raw_scores / SCORE_MAX

    # RAW metrics
    raw_mse_val = mse(pred_raw_scores, gold_raw_scores)
    raw_rmse_val = float(np.sqrt(raw_mse_val))
    raw_pearson, _ = pearsonr(pred_raw_scores, gold_raw_scores)
    raw_spearman, _ = spearmanr(pred_raw_scores, gold_raw_scores)
    raw_kendall, _ = kendalltau(pred_raw_scores, gold_raw_scores)

    # Normalized metrics
    norm_mse_val = mse(pred_norm_scores, gold_norm_scores)
    norm_rmse_val = float(np.sqrt(norm_mse_val))
    norm_pearson, _ = pearsonr(pred_norm_scores, gold_norm_scores)
    norm_spearman, _ = spearmanr(pred_norm_scores, gold_norm_scores)
    norm_kendall, _ = kendalltau(pred_norm_scores, gold_norm_scores)

    metrics = {
        "num_pairs_scored": int(len(pred_raw_scores)),
        "dataset": HF_DATASET_NAME,
        "model_name": model_name,
        "icl_n": int(icl_n),
        "raw": {
            "mse": float(raw_mse_val),
            "rmse": raw_rmse_val,
            "pearson_correlation": float(raw_pearson),
            "spearman_correlation": float(raw_spearman),
            "kendall_correlation": float(raw_kendall),
        },
        "normalized_0_1": {
            "mse": float(norm_mse_val),
            "rmse": norm_rmse_val,
            "pearson_correlation": float(norm_pearson),
            "spearman_correlation": float(norm_spearman),
            "kendall_correlation": float(norm_kendall),
        },
    }

    print("=======================================")
    print(f"Pairs evaluated: {metrics['num_pairs_scored']}")
    print("--- RAW score metrics (0-100) ---")
    print(f"Pearson r: {metrics['raw']['pearson_correlation']:.4f}")
    print(f"Spearman r: {metrics['raw']['spearman_correlation']:.4f}")
    print(f"Kendall tau: {metrics['raw']['kendall_correlation']:.4f}")
    print(f"MSE:  {metrics['raw']['mse']:.6f}")
    print(f"RMSE: {metrics['raw']['rmse']:.6f}")
    print("--- Normalized score metrics (0-1) ---")
    print(f"Pearson r: {metrics['normalized_0_1']['pearson_correlation']:.4f}")
    print(
        f"Spearman r: {metrics['normalized_0_1']['spearman_correlation']:.4f}"
    )
    print(
        f"Kendall tau: {metrics['normalized_0_1']['kendall_correlation']:.4f}"
    )
    print(f"MSE:  {metrics['normalized_0_1']['mse']:.6f}")
    print(f"RMSE: {metrics['normalized_0_1']['rmse']:.6f}")
    print(f"In-context examples used: {icl_n}")

    preds_path = os.path.join(out_dir, "predictions.csv")
    metrics_path = os.path.join(out_dir, "metrics.json")
    icl_path = os.path.join(out_dir, "icl_examples.json")

    with open(preds_path, "w", encoding="utf-8") as f:
        f.write(
            "idx,sentence1,sentence2,"
            "gold_score_raw_0_100,gold_score_norm_0_1,"
            "llm_score_raw_0_100,llm_score_norm_0_1,raw_model_reply\n"
        )
        # Save rows
        for r in rows_for_dump:

            def esc(x):
                return '"' + str(x).replace('"', '""') + '"'

            f.write(
                ",".join(
                    [
                        str(r["idx"]),
                        esc(r["sentence1"]),
                        esc(r["sentence2"]),
                        f"{r['gold_score_raw_0_100']:.3f}",
                        f"{r['gold_score_norm_0_1']:.6f}",
                        f"{r['llm_score_raw_0_100']:.3f}",
                        f"{r['llm_score_norm_0_1']:.6f}",
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
    parser = argparse.ArgumentParser(
        description="Evaluate LLM translation similarity on WMT20 SI-EN using 0–100 raw scores (and normalized 0–1)."
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        required=True,
        help="Base URL for the OpenAI-compatible API, e.g. http://localhost:8000/v1",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="Bearer token for the chat/completions endpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help='Model name to send in the "model" field (e.g. "Meta-Llama-3.1-8B-Instruct").',
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to write predictions.csv, metrics.json, and icl_examples.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only score this many test pairs (for quick runs).",
    )
    parser.add_argument(
        "--icl_n",
        type=int,
        default=0,
        help="How many random training examples to include as in-context demonstrations. "
        "0 means no in-context learning.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting in-context examples.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    run_eval(
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        out_dir=args.out_dir,
        limit=args.limit,
        icl_n=args.icl_n,
    )
