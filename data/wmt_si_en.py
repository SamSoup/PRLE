# data/wmt_sien.py

from .base import BaseRegressionDataModule
import datasets


class WMT20SIENDataModule(BaseRegressionDataModule):
    """
    samsoup/Samsoup-WMT2020-si-en DataModule

    Assumes the HF dataset has columns:
        - sentence1 (str)
        - sentence2 (str)
        - score     (float)

    And splits:
        - train
        - validation
        - test

    Behaves exactly like STSBDataModule / SICKRSTSDataModule:
      tokenize_inputs=False, combine_fields=True
        -> raw "text" = "s1 {sep} s2", labels from 'score'
      tokenize_inputs=False, combine_fields=False
        -> raw "text" = (s1, s2) tuple, labels from 'score'
      tokenize_inputs=True,  combine_fields=True
        -> tokenize single field "combined_text"; labels mapped to 'labels'
      tokenize_inputs=True,  combine_fields=False
        -> tokenize pair ("sentence1","sentence2"); labels mapped to 'labels'
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
        self.label_max = 100.0

    def setup(self, stage: str | None = None):
        # 1) load from our HF repo
        self.dataset = datasets.load_dataset("samsoup/Samsoup-WMT2020-si-en")

        if self.tokenize_inputs:
            # TOKENIZED path
            if self.combine_fields:
                # create single field
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

            # tokenize; base will map score -> labels
            self._tokenize_splits(remove_columns=[self.output_column])

        else:
            # RAW path
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

            # cast labels to torch
            for split in self.dataset:
                self.dataset[split].set_format(type=None)
                self.dataset[split].set_format(
                    type="torch",
                    columns=["labels"],
                    output_all_columns=True,
                )
                if "validation" in split:
                    self.eval_splits.append(split)
