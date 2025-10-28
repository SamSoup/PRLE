from .base import BaseRegressionDataModule
import datasets


class Yelp2018DataModule(BaseRegressionDataModule):
    """
    Yelp-2018 sentiment regression/classification with two fields per example:
      - text  : string review
      - label : int in [1..5]

    Modes (mirrors STSBDataModule behavior, minus combine_fields since it's single-string):
      1) tokenize_inputs=False  -> raw "text" passed downstream; labels standardized to "labels"
      2) tokenize_inputs=True   -> tokenize single field "text"; labels mapped to "labels"
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        tokenize_inputs: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            tokenize_inputs=tokenize_inputs,
            output_column="label",  # dataset's ground-truth column name
            **kwargs,
        )
        # Single-field input
        self.text_fields = ["text"]

    def setup(self, stage: str = None):
        # Load HF dataset: expected splits train/validation/test with sizes 7000/1500/1500
        self.dataset = datasets.load_dataset("samsoup/yelp2018")

        if self.tokenize_inputs:
            # TOKENIZED path: Base._tokenize() will convert source `self.output_column` -> "labels"
            # Use single-field tokenization over "text"
            self.text_fields = ["text"]
            self._tokenize_splits(remove_columns=[self.output_column])

        else:
            # RAW TEXT path: keep "text" as-is, standardize labels to "labels"
            for split in self.dataset:
                self.dataset[split] = self.dataset[split].map(
                    lambda x: {
                        "text": x["text"],
                        "labels": x[self.output_column],
                    }
                )

            # Cast numeric labels to torch; leave 'text' as Python objects
            for split in self.dataset:
                self.dataset[split].set_format(
                    type=None
                )  # clear prior formatting
                self.dataset[split].set_format(
                    type="torch", columns=["labels"], output_all_columns=True
                )
                if "validation" in split:
                    self.eval_splits.append(split)
