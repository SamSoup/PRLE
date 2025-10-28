# pip install datasets requests scipy numpy --quiet
#
# Usage example:
#   python sts_eval_llm.py \
#       --api_key YOUR_KEY_HERE \
#       --model_name Meta-Llama-3.1-8B-Instruct \
#       --out_dir ./results_run1 \
#       --limit 100 \
#       --icl_n 5
#
# What this script does:
#   1. Loads STS-B train split and test split.
#   2. Randomly samples N training examples (N = --icl_n) and uses them
#      as in-context demonstrations for EVERY test query.
#      Each demo includes two sentences and a human similarity score in [0.00, 1.00].
#   3. For each test pair, queries your chat/completions endpoint and asks
#      for a similarity score in [0.00, 1.00] with TWO digits after the decimal.
#   4. Collects model predictions, normalizes human gold labels to [0,1],
#      computes MSE and Pearson correlation.
#   5. Saves predictions (CSV) and metrics (JSON) to --out_dir.

import argparse
import os
import json
import time
import re
import requests
import numpy as np
import random
from datasets import load_dataset
from scipy.stats import pearsonr
from pprint import pprint
from tqdm.auto import tqdm

API_URL = "https://tejas.tacc.utexas.edu/v1/60709c21-409c-44a5-8a9d-1638fad5d5a6/chat/completions"

SYSTEM_MSG = (
    "You are an expert human annotator for semantic textual similarity (STS).\n"
    "You will see English sentence pairs and you must judge how similar in MEANING they are.\n\n"
    "Scoring instructions:\n"
    "- Output a real-valued similarity score from 0.00 to 1.00.\n"
    "- 1.00 = same meaning / paraphrases.\n"
    "- 0.50 = partially similar meaning but with important differences or missing info.\n"
    "- 0.00 = completely unrelated meaning.\n\n"
    "CRITICAL FORMAT RULE:\n"
    "Return ONLY the numeric score with EXACTLY two digits after the decimal point.\n"
    "Do NOT include any words, punctuation, units, labels, or explanation.\n"
)


def build_user_message(test_s1, test_s2, icl_examples):
    """
    Construct the user message sent to the LLM.
    If icl_examples is non-empty, prepend them as demonstrations.
    Each example includes Sentence A, Sentence B, and a gold similarity in [0.00, 1.00].
    """
    parts = []

    if icl_examples:
        parts.append("Here are some examples of how to score similarity:\n")
        for i, ex in enumerate(icl_examples, start=1):
            parts.append(
                f"Example {i}:\n"
                f"Sentence A: {ex['sentence1']}\n"
                f"Sentence B: {ex['sentence2']}\n"
                f"Similarity: {ex['score_0_1_two_decimals']}\n"
            )
        parts.append("\nNow score the new pair.\n")

    parts.append(
        "Sentence A: " + test_s1.strip() + "\n"
        "Sentence B: " + test_s2.strip() + "\n\n"
        "Answer with ONLY the similarity score in the range 0.00 to 1.00, "
        "using exactly two digits after the decimal point:"
    )

    return "\n".join(parts)


def query_similarity_llm(
    sentence_a, sentence_b, api_key, model_name, icl_examples, timeout=60
):
    """
    Ask the LLM for a 0.00-1.00 similarity score (two decimal places) for the given pair.
    Returns (score_float, raw_answer_str, full_json_dict).
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
        "max_tokens": 10,  # we expect something like "0.87"
        "temperature": 0.0,  # deterministic
    }

    resp = requests.post(
        API_URL, headers=headers, json=payload, timeout=timeout
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"LLM request failed: {resp.status_code} {resp.text}"
        )

    data = resp.json()

    # Prefer OpenAI-style field, fallback to .text (older style)
    raw_answer = None
    try:
        raw_answer = data["choices"][0]["message"]["content"]
    except (KeyError, TypeError):
        raw_answer = data["choices"][0].get("text")

    if raw_answer is None:
        raise RuntimeError(
            f"Could not find completion text in response:\n{data}"
        )

    # Parse a float between 0 and 1 with optional leading zero, like 0.87 or 1.00
    m = re.search(r"\b([01](?:\.\d+)?|\d?\.\d+)\b", raw_answer.strip())
    if not m:
        raise ValueError(
            f"Could not parse numeric similarity from: {raw_answer}"
        )

    score = float(m.group(0))
    score = max(0.0, min(1.0, score))  # clamp to [0,1]

    return score, raw_answer.strip(), data


def mse(pred, gold):
    """
    Mean Squared Error between two arrays of equal length.
    """
    pred = np.asarray(pred, dtype=float)
    gold = np.asarray(gold, dtype=float)
    return np.mean((pred - gold) ** 2)


def sample_icl_examples(train_split, n):
    """
    Randomly sample n examples from the STS-B train split and prepare them
    as few-shot / in-context learning demonstrations.
    Each example will store sentence1, sentence2, and the gold score normalized
    to [0,1] and formatted with two decimals.
    If n <= 0, returns [].
    """
    if n is None or n <= 0:
        return []

    # sample without replacement
    # note: train_split is a Dataset, we can index into it by integer
    idxs = random.sample(range(len(train_split)), k=min(n, len(train_split)))

    examples = []
    for idx in idxs:
        row = train_split[idx]
        s1 = row["sentence1"]
        s2 = row["sentence2"]
        score_norm = row["score"] / 5.0  # gold is [0,5]; normalize to [0,1]
        score_txt = f"{score_norm:.2f}"
        examples.append(
            {
                "sentence1": s1,
                "sentence2": s2,
                "score_0_1_two_decimals": score_txt,
            }
        )

    return examples


def run_eval(api_key, model_name, out_dir, limit, icl_n):
    # Make sure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset splits
    ds = load_dataset("sentence-transformers/stsb")
    stsb_train = ds["train"]
    stsb_test = ds["test"]

    # Prepare in-context examples once per run
    icl_examples = sample_icl_examples(stsb_train, icl_n)

    gold_scores = []
    pred_scores = []
    rows_for_dump = []

    for i, row in tqdm(enumerate(stsb_test), total=len(stsb_test)):
        if limit is not None and i >= limit:
            break

        s1 = row["sentence1"]
        s2 = row["sentence2"]

        # Human gold score in [0,5] -> normalize to [0,1]
        gold_norm = row["score"] / 5.0

        try:
            sim_score, raw_answer, raw_json = query_similarity_llm(
                s1,
                s2,
                api_key=api_key,
                model_name=model_name,
                icl_examples=icl_examples,
            )
        except Exception as e:
            print(f"[{i}] ERROR querying model: {e}")
            continue

        gold_scores.append(gold_norm)
        pred_scores.append(sim_score)

        rows_for_dump.append(
            {
                "idx": i,
                "sentence1": s1,
                "sentence2": s2,
                "gold_similarity_0_1": gold_norm,
                "llm_similarity_0_1": sim_score,
                "raw_model_reply": raw_answer,
            }
        )

        if i % 100 == 0:
            print(f"[{i}]")
            print("Sentence 1:", s1)
            print("Sentence 2:", s2)
            print("Gold (0-1):", f"{gold_norm:.2f}")
            print("LLM  (0-1):", f"{sim_score:.2f}")
            print("Raw model reply:", raw_answer)
            print()
            time.sleep(0.25)

    gold_scores = np.array(gold_scores, dtype=float)
    pred_scores = np.array(pred_scores, dtype=float)

    # Metrics
    pearson_r, _ = pearsonr(pred_scores, gold_scores)
    mse_val = mse(pred_scores, gold_scores)

    metrics = {
        "num_pairs_scored": int(len(pred_scores)),
        "pearson_correlation": float(pearson_r),
        "mse": float(mse_val),
        "model_name": model_name,
        "icl_n": int(icl_n),
    }

    print("=======================================")
    print(f"Pairs evaluated: {metrics['num_pairs_scored']}")
    print(
        f"Pearson r (LLM vs human, 0-1 scale): {metrics['pearson_correlation']:.4f}"
    )
    print(f"MSE       (LLM vs human, 0-1 scale): {metrics['mse']:.6f}")
    print(f"In-context examples used: {icl_n}")

    # Save outputs
    preds_path = os.path.join(out_dir, "predictions.csv")
    metrics_path = os.path.join(out_dir, "metrics.json")
    icl_path = os.path.join(out_dir, "icl_examples.json")

    # Save predictions as CSV
    with open(preds_path, "w", encoding="utf-8") as f:
        f.write(
            "idx,sentence1,sentence2,gold_similarity_0_1,llm_similarity_0_1,raw_model_reply\n"
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
                        f"{r['gold_similarity_0_1']:.6f}",
                        f"{r['llm_similarity_0_1']:.6f}",
                        esc(r["raw_model_reply"]),
                    ]
                )
                + "\n"
            )

    # Save metrics as JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save the actual ICL examples we used for reproducibility
    with open(icl_path, "w", encoding="utf-8") as f:
        json.dump(icl_examples, f, indent=2)

    print(f"\nSaved predictions to: {preds_path}")
    print(f"Saved metrics to:     {metrics_path}")
    print(f"Saved ICL examples to:{icl_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM sentence similarity on STS-B using 0.00-1.00 scores."
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
        api_key=args.api_key,
        model_name=args.model_name,
        out_dir=args.out_dir,
        limit=args.limit,
        icl_n=args.icl_n,
    )
