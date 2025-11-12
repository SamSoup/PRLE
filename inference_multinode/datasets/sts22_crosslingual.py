from __future__ import annotations
import random
import re
from typing import List, Tuple, Dict, Any
from datasets import load_dataset

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


def build_user_message(
    test_s1: str, test_s2: str, icl_examples: List[Dict[str, str]]
) -> str:
    parts: List[str] = []
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


class InferenceModule:
    """STS22 cross-lingual (labels in 1..4; we return gold normalized to 0..1)."""

    def __init__(self, *, limit: int | None = None, seed: int = 42):
        self.limit = limit
        self.seed = seed

    def _sample_icl_examples(self, train_split, n: int) -> List[Dict[str, str]]:
        if n is None or n <= 0 or train_split is None or len(train_split) == 0:
            return []
        rng = random.Random(self.seed)
        idxs = rng.sample(range(len(train_split)), k=min(n, len(train_split)))
        exs: List[Dict[str, str]] = []
        for i in idxs:
            row = train_split[i]
            s1 = row["sentence1"]
            s2 = row["sentence2"]
            score_raw = float(row["score"])  # 1..4
            exs.append(
                {
                    "sentence1": s1,
                    "sentence2": s2,
                    "score_1_4_up_to_3": f"{score_raw:.3f}",
                }
            )
        return exs

    def prepare(
        self, *, icl_n: int
    ) -> tuple[list[tuple[str, str, float]], list[float], list[dict[str, str]]]:
        ds = load_dataset("Samsoup/sts22-crosslingual-sts")
        train = ds.get("train")  # may be None/missing
        test = ds["test"]
        if self.limit is not None:
            test = test.select(range(min(self.limit, len(test))))
        icl_examples = self._sample_icl_examples(train, icl_n)

        triplets: list[tuple[str, str, float]] = []
        golds: list[float] = []
        for row in test:
            s1 = row["sentence1"]
            s2 = row["sentence2"]
            gold_raw = float(row["score"])  # 1..4
            gold01 = (gold_raw - 1.0) / 3.0
            gold01 = max(0.0, min(1.0, gold01))
            triplets.append((s1, s2, gold01))
            golds.append(gold01)
        return triplets, golds, icl_examples


def parse_reply_to_0_1(raw: str) -> float:
    # Expect 1..4 with up to 3 decimals → normalize to [0,1]
    m = re.search(r"\b\d+(?:\.\d+)?\b", raw.strip())
    v = 2.5 if not m else float(m.group(0))
    v = max(1.0, min(4.0, v))
    return float((v - 1.0) / 3.0)
