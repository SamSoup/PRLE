# /data/base.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch


class BaseRegressionDataModule(pl.LightningDataModule):
    """
    Generic regression DataModule that can:
      - tokenize (HuggingFace tokenizer) OR pass raw text downstream
      - handle single-field or pair-field tokenization (via self.text_fields set in subclass/setup)

    Subclasses must:
      - load `self.dataset` with splits: train/validation/test
      - define `self.text_fields`:
          * ["combined_text"]           -> single string per example
          * ["sentence1", "sentence2"]  -> a pair per example
      - call `_tokenize_splits(remove_columns=[...])` if tokenize_inputs=True
      - OR create raw fields: "text" (string or (string,string)) and the label column
        indicated by `self.output_column` (default: "label"). Collation will emit "labels".
    """

    # Columns used when tokenizing with HF tokenizer
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        tokenize_inputs: bool = True,
        output_column: str = "label",  # <-- name of label column in the source dataset
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenize_inputs = tokenize_inputs
        self.output_column = output_column

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name_or_path, use_fast=True
            )
            if self.tokenize_inputs
            else None
        )

        self.dataset = None
        self.columns = None
        self.eval_splits = []
        # Subclass should set this during setup() to either:
        #   ["combined_text"] OR ["sentence1", "sentence2"]
        self.text_fields = None

    def setup(self, stage: str = None):
        raise NotImplementedError(
            "Each dataset-specific DataModule must override `setup()`"
        )

    # -------- dataloaders --------
    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=None if self.tokenize_inputs else self._collate_raw,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.eval_batch_size,
            collate_fn=None if self.tokenize_inputs else self._collate_raw,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.eval_batch_size,
            collate_fn=None if self.tokenize_inputs else self._collate_raw,
        )

    # -------- helpers --------
    def _collate_raw(self, batch):
        """
        Collate raw text examples (no tokenization).
        Expects each item to have:
          - "text": str OR tuple(str, str)
          - label under either "labels" or `self.output_column`
        Returns:
          { "text": List[str] or List[Tuple[str,str]], "labels": Tensor[B] }
        """
        texts = [ex["text"] for ex in batch]
        # accept either normalized "labels" or original dataset column name
        labels_list = []
        for ex in batch:
            if "labels" in ex:
                labels_list.append(float(ex["labels"]))
            else:
                labels_list.append(float(ex[self.output_column]))
        labels = torch.tensor(labels_list, dtype=torch.float32)
        return {"text": texts, "labels": labels}

    def _tokenize_splits(self, remove_columns):
        """
        Tokenize using self.text_fields:
          - If len(self.text_fields) == 1: tokenizes a single string field (e.g., "combined_text")
          - If len(self.text_fields) == 2: tokenizes a pair (sentence1, sentence2)
        """
        if not self.tokenize_inputs:
            raise RuntimeError(
                "Called _tokenize_splits while tokenize_inputs=False."
            )
        if not self.text_fields:
            raise RuntimeError(
                "self.text_fields must be set before calling _tokenize_splits()."
            )
        if len(self.text_fields) not in (1, 2):
            raise ValueError("self.text_fields must have length 1 or 2.")

        for split in self.dataset.keys():
            print(f"Processing split: {split}")
            self.dataset[split] = self.dataset[split].map(
                self._tokenize, batched=True, remove_columns=remove_columns
            )
            self.columns = [
                c
                for c in self.dataset[split].column_names
                if c in self.loader_columns
            ]
            print(f"Using columns: {self.columns}")
            self.dataset[split].set_format(type="torch", columns=self.columns)
            if "validation" in split:
                self.eval_splits.append(split)

    def _tokenize(self, example_batch, indices=None):
        # Build batch as either a list[str] or a list[Tuple[str,str]]
        if len(self.text_fields) == 1:
            texts_or_text_pairs = example_batch[self.text_fields[0]]
        else:
            texts_or_text_pairs = list(
                zip(
                    example_batch[self.text_fields[0]],
                    example_batch[self.text_fields[1]],
                )
            )

        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
        )
        # Normalize the dataset's label column name -> "labels"
        features["labels"] = example_batch[self.output_column]
        return features
