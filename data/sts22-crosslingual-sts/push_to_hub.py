"""
Process the default subset of `mteb/sts22-crosslingual-sts` and push to
`samsoup/sts22-crosslingual-sts`.

Steps:
  - load_dataset("mteb/sts22-crosslingual-sts", name="default")
  - dataset already has train / test
  - from TRAIN: shuffle (deterministic) and carve out 30% → validation
  - keep ONLY: sentence1, sentence2, score
  - push to HF as train / validation / test

Usage:
  pip install datasets huggingface_hub
  huggingface-cli login
  python push_sts22_crosslingual_sts.py --repo samsoup/sts22-crosslingual-sts
"""

import argparse
import os
from datetime import datetime
from typing import List

from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi


# we normalize to these 3:
#   sentence1, sentence2, score
CANDIDATE_SENT1 = ["sentence1", "sentence_A", "text1", "src", "source"]
CANDIDATE_SENT2 = ["sentence2", "sentence_B", "text2", "tgt", "target"]
CANDIDATE_SCORE = ["score", "label", "similarity", "relatedness_score"]


def _first_present(cols: List[str], candidates: List[str]) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"None of {candidates} found in columns {cols}")


def load_and_split(seed: int = 42) -> DatasetDict:
    # 1) load default subset
    # if HF ever changes the name, this is the place to tweak
    raw = load_dataset("mteb/sts22-crosslingual-sts", name="default")

    if "train" not in raw or "test" not in raw:
        raise RuntimeError(
            f"Expected 'train' and 'test' in mteb/sts22-crosslingual-sts (default), got {list(raw.keys())}"
        )

    ds_train = raw["train"]
    ds_test = raw["test"]

    # figure out current column names from train
    cols = ds_train.column_names
    col_s1 = _first_present(cols, CANDIDATE_SENT1)
    col_s2 = _first_present(cols, CANDIDATE_SENT2)
    col_y = _first_present(cols, CANDIDATE_SCORE)

    # 2) shuffle train deterministically, then carve 30% → val
    ds_train = ds_train.shuffle(seed=seed)
    n = len(ds_train)
    n_val = int(0.3 * n)
    n_train = n - n_val

    ds_train_new = ds_train.select(range(0, n_train))
    ds_val_new = ds_train.select(range(n_train, n))

    # 3) normalize to {sentence1, sentence2, score} and drop extra columns
    features = Features(
        {
            "sentence1": Value("string"),
            "sentence2": Value("string"),
            "score": Value("float32"),
        }
    )

    def _map_to_canonical(ex):
        return {
            "sentence1": ex[col_s1],
            "sentence2": ex[col_s2],
            "score": round(float(ex[col_y]), 3),
        }

    ds_train_new = ds_train_new.map(_map_to_canonical, remove_columns=cols)
    ds_val_new = ds_val_new.map(_map_to_canonical, remove_columns=cols)

    # we ALSO normalize test in the same way — test may have the same or slightly different columns
    test_cols = ds_test.column_names
    test_s1 = _first_present(test_cols, CANDIDATE_SENT1)
    test_s2 = _first_present(test_cols, CANDIDATE_SENT2)
    test_y = _first_present(test_cols, CANDIDATE_SCORE)

    def _map_test(ex):
        return {
            "sentence1": ex[test_s1],
            "sentence2": ex[test_s2],
            "score": float(ex[test_y]),
        }

    ds_test_new = ds_test.map(_map_test, remove_columns=test_cols)

    # rebuild with explicit features so the hub has a clean schema
    ds_train_new = Dataset.from_dict(ds_train_new.to_dict(), features=features)
    ds_val_new = Dataset.from_dict(ds_val_new.to_dict(), features=features)
    ds_test_new = Dataset.from_dict(ds_test_new.to_dict(), features=features)

    return DatasetDict(
        train=ds_train_new,
        validation=ds_val_new,
        test=ds_test_new,
    )


def make_readme(n_train: int, n_val: int, n_test: int) -> str:
    today = datetime.utcnow().date().isoformat()
    return f"""---
dataset_info:
  features:
  - name: sentence1
    dtype: string
  - name: sentence2
    dtype: string
  - name: score
    dtype: float32
  splits:
  - name: train
    num_examples: {n_train}
  - name: validation
    num_examples: {n_val}
  - name: test
    num_examples: {n_test}
  task_categories:
  - text-regression
  task_ids:
  - semantic-textual-similarity
---

# samsoup/sts22-crosslingual-sts

This dataset is derived from the **default subset** of
`mteb/sts22-crosslingual-sts`.

The original dataset provides **train** and **test**. We:

1. loaded the default subset,
2. took the original **train** split, shuffled it deterministically,
3. split that into **train (70%)** and **validation (30%)**,
4. normalized columns to
   - `sentence1`
   - `sentence2`
   - `score` (float32),
5. kept the original **test** (but normalized columns the same way),
6. pushed as `train` / `validation` / `test`.

This makes the dataset easy to drop into STS-style regression training pipelines.

**Fields**
- `sentence1`: first sentence (string)
- `sentence2`: second sentence (string)
- `score`: cross-lingual similarity score (float32)

**Notes**
- This is a *cross-lingual* STS dataset.
- The 70/30 split of original-train is deterministic (seed=42 by default).
- Test size is whatever the upstream MTEB dataset provides.

_Last updated: {today}_
"""


def push_to_hub(dset: DatasetDict, repo_id: str, token: str | None):
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    n_train = len(dset["train"])
    n_val = len(dset["validation"])
    n_test = len(dset["test"])

    dset.push_to_hub(
        repo_id,
        private=False,
        token=token,
        commit_message="Initial push of sts22-crosslingual-sts (train→train/val, normalized columns)",
    )

    api.upload_file(
        path_or_fileobj=make_readme(n_train, n_val, n_test).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card for samsoup/sts22-crosslingual-sts",
    )

    print(f"Pushed to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="Samsoup/sts22-crosslingual-sts",
        help="Target HF dataset repo",
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token or use CLI login",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for train split before 70/30 partition",
    )
    args = parser.parse_args()

    dset = load_and_split(seed=args.seed)
    print(dset)

    push_to_hub(dset, repo_id=args.repo, token=args.hf_token)


if __name__ == "__main__":
    main()
