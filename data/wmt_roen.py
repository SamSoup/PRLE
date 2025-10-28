# /data/wmt2020roen.py

from .base import BaseRegressionDataModule
import datasets


class WMT20ROENDataModule(BaseRegressionDataModule):
    """
    WMT20 MLQE Task 1 (ro-en) regression DataModule.

    Matches the same behaviors as STSBDataModule (see en-zh module).
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
            output_column="mean",
            **kwargs,
        )
        self.combine_fields = combine_fields
        self.combine_separator_token = combine_separator_token
        self.src_lang = "ro"
        self.tgt_lang = "en"

    def setup(self, stage: str = None):
        self.dataset = datasets.load_dataset("wmt/wmt20_mlqe_task1", "ro-en")

        if self.tokenize_inputs:
            if self.combine_fields:
                for split in self.dataset:
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "combined_text": x["translation"][self.src_lang]
                            + f" {self.combine_separator_token} "
                            + x["translation"][self.tgt_lang],
                            self.output_column: x[self.output_column],
                        }
                    )
                self.text_fields = ["combined_text"]
            else:
                for split in self.dataset:
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "source": x["translation"][self.src_lang],
                            "target": x["translation"][self.tgt_lang],
                            self.output_column: x[self.output_column],
                        }
                    )
                self.text_fields = ["source", "target"]

            self._tokenize_splits(remove_columns=[self.output_column])

        else:
            for split in self.dataset:
                if self.combine_fields:
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "text": x["translation"][self.src_lang]
                            + f" {self.combine_separator_token} "
                            + x["translation"][self.tgt_lang],
                            "labels": x[self.output_column],
                        }
                    )
                else:
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "text": (
                                x["translation"][self.src_lang],
                                x["translation"][self.tgt_lang],
                            ),
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
