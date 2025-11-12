from __future__ import annotations
import random
import re
from typing import List, Tuple, Dict, Any
from datasets import load_dataset

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


def build_user_message(
    test_s1: str, test_s2: str, icl_examples: List[Dict[str, str]]
) -> str:
    parts: List[str] = []
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
        "Sentence A: "
        + test_s1.strip()
        + "\nSentence B: "
        + test_s2.strip()
        + "\n\nAnswer with ONLY the similarity score in the range 0.00 to 1.00, "
        "using exactly two digits after the decimal point:"
    )
    return "\n".join(parts)


class InferenceModule:
    """STS-B (labels in 0..1 after normalization)."""

    def __init__(self, *, limit: int | None = None, seed: int = 42):
        self.limit = limit
        self.seed = seed

    def _sample_icl_examples(self, train_split, n: int) -> List[Dict[str, str]]:
        if n is None or n <= 0:
            return []
        rng = random.Random(self.seed)
        idxs = rng.sample(range(len(train_split)), k=min(n, len(train_split)))
        exs: List[Dict[str, str]] = []
        for i in idxs:
            row = train_split[i]
            s1 = row["sentence1"]
            s2 = row["sentence2"]
            score_norm = float(row["score"]) / 5.0
            exs.append(
                {
                    "sentence1": s1,
                    "sentence2": s2,
                    "score_0_1_two_decimals": f"{score_norm:.2f}",
                }
            )
        return exs

    def prepare(
        self, *, icl_n: int
    ) -> tuple[list[tuple[str, str, float]], list[float], list[dict[str, str]]]:
        ds = load_dataset("sentence-transformers/stsb")
        train = ds["train"]
        test = ds["test"]
        if self.limit is not None:
            test = test.select(range(min(self.limit, len(test))))
        icl_examples = self._sample_icl_examples(train, icl_n)

        triplets: list[tuple[str, str, float]] = []
        golds: list[float] = []
        for row in test:
            s1 = row["sentence1"]
            s2 = row["sentence2"]
            gold01 = float(row["score"]) / 5.0
            gold01 = max(0.0, min(1.0, gold01))
            triplets.append((s1, s2, gold01))
            golds.append(gold01)
        return triplets, golds, icl_examples


def parse_reply_to_0_1(raw: str) -> float:
    # Lenient but safe: accept 0, 1, 0.x, .x, 0.xx, 1.0, 1.00; reject 10.0 etc.
    m = re.search(
        r"(?<!\d)(?:"
        r"0(?:\.\d{1,2})?"  # 0, 0.x, 0.xx
        r"|1(?:\.0{1,2})?"  # 1, 1.0, 1.00
        r"|\.\d{1,2}"  # .x, .xx
        r")(?!\d)",
        raw.strip(),
    )
    val = float(m.group(0)) if m else 0.0
    return max(0.0, min(1.0, val))
