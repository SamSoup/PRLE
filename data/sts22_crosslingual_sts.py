# data/sts22_crosslingual_sts.py

from .base import BaseRegressionDataModule
import datasets


class STS22CrosslingualSTSDataModule(BaseRegressionDataModule):
    """
    samsoup/sts22-crosslingual-sts DataModule

    Assumes columns:
        - sentence1
        - sentence2
        - score

    And splits (as we pushed them):
        - train
        - validation   (30% of original train)
        - test         (original test, normalized)

    Same behavior/options as STSBDataModule.
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
        # 1) load from our HF repo
        self.dataset = datasets.load_dataset("Samsoup/sts22-crosslingual-sts")

        if self.tokenize_inputs:
            if self.combine_fields:
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
                self.text_fields = ["sentence1", "sentence2"]

            self._tokenize_splits(remove_columns=[self.output_column])

        else:
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

            for split in self.dataset:
                self.dataset[split].set_format(type=None)
                self.dataset[split].set_format(
                    type="torch", columns=["labels"], output_all_columns=True
                )
                if "validation" in split:
                    self.eval_splits.append(split)
