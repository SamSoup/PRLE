# /data/stsb.py

from .base import BaseRegressionDataModule
import datasets


class STSBDataModule(BaseRegressionDataModule):
    """
    STS-B DataModule with four modes:
      1) tokenize_inputs=False, combine_fields=True   -> raw "text" = "s1 {sep} s2", labels from 'score'
      2) tokenize_inputs=False, combine_fields=False  -> raw "text" = (s1, s2) tuple, labels from 'score'
      3) tokenize_inputs=True,  combine_fields=True   -> tokenize single field "combined_text"; labels mapped to 'labels'
      4) tokenize_inputs=True,  combine_fields=False  -> tokenize pair ("sentence1","sentence2"); labels mapped to 'labels'
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        tokenize_inputs: bool = True,
        combine_fields: bool = False,
        combine_separator_token: str = "[SEP]",  # e.g., can switch to "\n"
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            tokenize_inputs=tokenize_inputs,
            output_column="score",  # <-- STSB ground-truth column
            **kwargs,
        )
        self.combine_fields = combine_fields
        self.combine_separator_token = combine_separator_token

    def setup(self, stage: str = None):
        # Load the HF dataset again to override
        self.dataset = datasets.load_dataset("sentence-transformers/stsb")

        if self.tokenize_inputs:
            # TOKENIZED paths (Base._tokenize() will convert source `self.output_column` -> "labels")
            if self.combine_fields:
                # Create a single combined field to tokenize (retain the original score column!)
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
                # Use sentence pair tokenization
                self.text_fields = ["sentence1", "sentence2"]

            # Tokenize and map `self.output_column` -> "labels"
            self._tokenize_splits(remove_columns=[self.output_column])

        else:
            # RAW TEXT paths (no tokenization) -> standardize labels to "labels"
            for split in self.dataset:
                if self.combine_fields:
                    # Single string with separator
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "text": x["sentence1"]
                            + f" {self.combine_separator_token} "
                            + x["sentence2"],
                            "labels": x[self.output_column],
                        }
                    )
                else:
                    # Tuple of strings for pairwise SentenceEncoder features
                    self.dataset[split] = self.dataset[split].map(
                        lambda x: {
                            "text": (x["sentence1"], x["sentence2"]),
                            "labels": x[self.output_column],
                        }
                    )

            # Only cast numeric labels to torch; leave 'text' as Python objects
            for split in self.dataset:
                self.dataset[split].set_format(
                    type=None
                )  # clear any prior formatting
                self.dataset[split].set_format(
                    type="torch", columns=["labels"], output_all_columns=True
                )
                if "validation" in split:
                    self.eval_splits.append(split)
