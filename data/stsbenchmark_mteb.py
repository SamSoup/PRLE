# data/stsbenchmark_mteb.py

import datasets
from .base import BaseRegressionDataModule


class STSBenchmarkMTEBDataModule(BaseRegressionDataModule):
    """
    DataModule for mteb/stsbenchmark-sts

    Assumptions:
      - columns: sentence1, sentence2, score
      - splits: train, validation, test
      - score is UNnormalized (typically 0..5), unlike sentence-transformers/stsb (0..1)

    Modes (same as STSBDataModule):
      1) tokenize_inputs=False, combine_fields=True
         -> raw "text" = "s1 {sep} s2", labels from 'score'
      2) tokenize_inputs=False, combine_fields=False
         -> raw "text" = (s1, s2), labels from 'score'
      3) tokenize_inputs=True, combine_fields=True
         -> tokenize "combined_text", labels -> "labels"
      4) tokenize_inputs=True, combine_fields=False
         -> tokenize pair ("sentence1","sentence2"), labels -> "labels"
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        tokenize_inputs: bool = True,
        combine_fields: bool = False,
        combine_separator_token: str = "[SEP]",
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            tokenize_inputs=tokenize_inputs,
            output_column="score",
            **kwargs,
        )
        self.combine_fields = combine_fields
        self.combine_separator_token = combine_separator_token

    def setup(self, stage: str = None):
        # load the MTEB version
        self.dataset = datasets.load_dataset("mteb/stsbenchmark-sts")

        if self.tokenize_inputs:
            # TOKENIZED path
            if self.combine_fields:
                # create a single string field
                for split in self.dataset:
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "combined_text": x["sentence1"]
                            + f" {self.combine_separator_token} "
                            + x["sentence2"],
                            self.output_column: x[self.output_column],
                        }
                    )
                self.text_fields = ["combined_text"]
            else:
                # standard pair
                self.text_fields = ["sentence1", "sentence2"]

            # tokenize + map score -> labels
            self._tokenize_splits(remove_columns=[self.output_column])

        else:
            # RAW path (no tokenization here)
            for split in self.dataset:
                if self.combine_fields:
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "text": x["sentence1"]
                            + f" {self.combine_separator_token} "
                            + x["sentence2"],
                            "labels": x[self.output_column],
                        }
                    )
                else:
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "text": (x["sentence1"], x["sentence2"]),
                            "labels": x[self.output_column],
                        }
                    )

            # cast labels to torch tensors
            for split in self.dataset:
                self.dataset[split].set_format(type=None)
                self.dataset[split].set_format(
                    type="torch", columns=["labels"], output_all_columns=True
                )
                if "validation" in split:
                    self.eval_splits.append(split)
