# pip install datasets requests scipy numpy --quiet
#
# Usage example:
#   python sts_eval_llm_mteb_sickr.py \
#       --api_key YOUR_KEY_HERE \
#       --model_name Meta-Llama-3.1-8B-Instruct \
#       --out_dir ./results_mteb_sickr \
#       --limit 200 \
#       --icl_n 5
#
# Notes:
#   - Samsoup/sickr-sts may not include a train split. If no train split,
#     ICL is disabled automatically regardless of --icl_n.
#   - Labels are in [1,5]. We ask the model for a number in [1,5] with up to 3 decimals.
#   - We now report metrics on BOTH scales:
#       * raw 1..5
#       * normalized 0..1 via (x - 1) / 4

import argparse
import os
import json
import re
import requests
import numpy as np
import random
from datasets import load_dataset
from scipy.stats import pearsonr
from tqdm.auto import tqdm

API_URL = "https://tejas.tacc.utexas.edu/v1/60709c21-409c-44a5-8a9d-1638fad5d5a6/chat/completions"

SYSTEM_MSG = (
    "You are an expert annotator for semantic textual similarity (STS).\n"
    "You will see sentence pairs and judge how similar in MEANING they are.\n\n"
    "Scoring instructions:\n"
    "- Output a real-valued similarity score from 1 to 5, inclusive.\n"
    "- 5 = same meaning / paraphrases; 1 = completely unrelated.\n\n"
    "CRITICAL FORMAT RULE:\n"
    "Return ONLY the numeric score, with up to three digits after the decimal (e.g., 3, 3.1, 3.14, 3.142).\n"
    "Do NOT include any words, punctuation, units, labels, or explanation.\n"
)


def build_user_message(test_s1, test_s2, icl_examples):
    parts = []
    if icl_examples:
        parts.append("Here are some examples of how to score similarity:\n")
        for i, ex in enumerate(icl_examples, start=1):
            parts.append(
                f"Example {i}:\n"
                f"Sentence A: {ex['sentence1']}\n"
                f"Sentence B: {ex['sentence2']}\n"
                f"Similarity (1..5): {ex['score_1_5_up_to_3']}\n"
            )
        parts.append("\nNow score the new pair.\n")
    parts.append(
        "Sentence A: " + test_s1.strip() + "\n"
        "Sentence B: " + test_s2.strip() + "\n\n"
        "Answer with ONLY the similarity score in the range 1..5 (inclusive), "
        "using up to three digits after the decimal:"
    )
    return "\n".join(parts)


def query_similarity_llm(
    sentence_a, sentence_b, api_key, model_name, icl_examples, timeout=60
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
    resp = requests.post(
        API_URL, headers=headers, json=payload, timeout=timeout
    )
    if resp.status_code != 200:
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

    m = re.search(r"\b\d+(?:\.\d+)?\b", raw_answer.strip())
    if not m:
        raise ValueError(
            f"Could not parse numeric similarity from: {raw_answer}"
        )
    score = float(m.group(0))
    score = max(1.0, min(5.0, score))
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
        score_raw = float(row["score"])  # 1..5
        score_txt = f"{score_raw:.3f}"
        examples.append(
            {"sentence1": s1, "sentence2": s2, "score_1_5_up_to_3": score_txt}
        )
    return examples


def run_eval(api_key, model_name, out_dir, limit, icl_n, seed):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    ds = load_dataset("Samsoup/sickr-sts")
    train_split = ds.get("train", None)
    test_split = ds["test"]

    icl_examples = sample_icl_examples(train_split, icl_n)

    # store raw & normalized
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
        gold_raw = float(row["score"])  # 1..5
        gold_norm = (gold_raw - 1.0) / 4.0  # → [0,1]

        try:
            pred_raw, raw_answer, _ = query_similarity_llm(
                s1,
                s2,
                api_key=api_key,
                model_name=model_name,
                icl_examples=icl_examples,
            )
        except Exception as e:
            print(f"[{i}] ERROR querying model: {e}")
            continue

        pred_norm = (pred_raw - 1.0) / 4.0

        gold_scores_raw.append(gold_raw)
        pred_scores_raw.append(pred_raw)
        gold_scores_norm.append(gold_norm)
        pred_scores_norm.append(pred_norm)

        rows_for_dump.append(
            {
                "idx": i,
                "sentence1": s1,
                "sentence2": s2,
                "gold_similarity_1_5": float(f"{gold_raw:.3f}"),
                "llm_similarity_1_5": float(f"{pred_raw:.3f}"),
                "gold_similarity_0_1": float(f"{gold_norm:.6f}"),
                "llm_similarity_0_1": float(f"{pred_norm:.6f}"),
                "raw_model_reply": raw_answer,
            }
        )

        if i % 100 == 0:
            print(f"[{i}] Gold 1..5: {gold_raw:.3f} | LLM 1..5: {pred_raw:.3f}")

    # convert to np
    gold_scores_raw = np.array(gold_scores_raw, dtype=float)
    pred_scores_raw = np.array(pred_scores_raw, dtype=float)
    gold_scores_norm = np.array(gold_scores_norm, dtype=float)
    pred_scores_norm = np.array(pred_scores_norm, dtype=float)

    # metrics on RAW 1..5
    pearson_raw, _ = pearsonr(pred_scores_raw, gold_scores_raw)
    mse_raw = mse(pred_scores_raw, gold_scores_raw)

    # metrics on NORMALIZED 0..1
    pearson_norm, _ = pearsonr(pred_scores_norm, gold_scores_norm)
    mse_norm = mse(pred_scores_norm, gold_scores_norm)

    metrics = {
        "num_pairs_scored": int(len(pred_scores_raw)),
        # raw scale
        "pearson_correlation_raw": float(pearson_raw),
        "mse_raw": float(mse_raw),
        # normalized scale
        "pearson_correlation_norm": float(pearson_norm),
        "mse_norm": float(mse_norm),
        "model_name": model_name,
        "icl_n": int(len(icl_examples)),
        "label_range": "1..5 (also reported normalized to 0..1 via (x-1)/4)",
    }

    print("=======================================")
    print(f"Pairs evaluated: {metrics['num_pairs_scored']}")
    print(f"RAW 1..5     → Pearson: {metrics['pearson_correlation_raw']:.4f}")
    print(f"RAW 1..5     → MSE:     {metrics['mse_raw']:.6f}")
    print(f"NORMALIZED   → Pearson: {metrics['pearson_correlation_norm']:.4f}")
    print(f"NORMALIZED   → MSE:     {metrics['mse_norm']:.6f}")
    print(f"In-context examples used: {metrics['icl_n']}")

    preds_path = os.path.join(out_dir, "predictions.csv")
    metrics_path = os.path.join(out_dir, "metrics.json")
    icl_path = os.path.join(out_dir, "icl_examples.json")

    with open(preds_path, "w", encoding="utf-8") as f:
        f.write(
            "idx,sentence1,sentence2,gold_similarity_1_5,llm_similarity_1_5,"
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
                        f"{r['gold_similarity_1_5']:.3f}",
                        f"{r['llm_similarity_1_5']:.3f}",
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
        description="Evaluate LLM similarity on Samsoup/sickr-sts (1..5 labels)."
    )
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
        args.api_key,
        args.model_name,
        args.out_dir,
        args.limit,
        args.icl_n,
        args.seed,
    )
