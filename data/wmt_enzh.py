# /data/wmt2020enzh.py

from .base import BaseRegressionDataModule
import datasets


class WMT20ENZHDataModule(BaseRegressionDataModule):
    """
    WMT20 MLQE Task 1 (en-zh) regression DataModule.

    Behavior matches STSBDataModule:
      - tokenize_inputs=False, combine_fields=True   -> raw "text" = "src {sep} tgt", labels from 'mean'
      - tokenize_inputs=False, combine_fields=False  -> raw "text" = (src, tgt), labels from 'mean'
      - tokenize_inputs=True,  combine_fields=True   -> tokenize single field "combined_text"; labels mapped to 'labels'
      - tokenize_inputs=True,  combine_fields=False  -> tokenize pair ("source","target"); labels mapped to 'labels'
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
            output_column="mean",  # <-- MLQE label column
            **kwargs,
        )
        self.combine_fields = combine_fields
        self.combine_separator_token = combine_separator_token
        self.src_lang = "en"
        self.tgt_lang = "zh"

    def setup(self, stage: str = None):
        # Load the HF dataset subset: en-zh
        self.dataset = datasets.load_dataset("wmt/wmt20_mlqe_task1", "en-zh")

        if self.tokenize_inputs:
            # TOKENIZED paths (Base._tokenize() maps `self.output_column` -> "labels")
            if self.combine_fields:
                # Build a single combined field to tokenize
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
                # Use sentence pair tokenization from extracted fields
                for split in self.dataset:
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "source": x["translation"][self.src_lang],
                            "target": x["translation"][self.tgt_lang],
                            self.output_column: x[self.output_column],
                        }
                    )
                self.text_fields = ["source", "target"]

            # Tokenize and map `self.output_column` -> "labels"
            self._tokenize_splits(remove_columns=[self.output_column])

        else:
            # RAW TEXT paths (no tokenization) -> standardize labels to "labels"
            for split in self.dataset:
                if self.combine_fields:
                    # Single string with separator
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "text": x["translation"][self.src_lang]
                            + f" {self.combine_separator_token} "
                            + x["translation"][self.tgt_lang],
                            "labels": x[self.output_column],
                        }
                    )
                else:
                    # Tuple of strings for pair features
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "text": (
                                x["translation"][self.src_lang],
                                x["translation"][self.tgt_lang],
                            ),
                            "labels": x[self.output_column],
                        }
                    )

            # Only cast numeric labels to torch; leave 'text' as Python objects
            for split in self.dataset:
                self.dataset[split].set_format(type=None)
                self.dataset[split].set_format(
                    type="torch", columns=["labels"], output_all_columns=True
                )
                if "validation" in split:
                    self.eval_splits.append(split)
