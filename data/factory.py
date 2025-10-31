# data/factory.py

from .stsb import STSBDataModule
from .yelp import Yelp2018DataModule
from .wmt_enzh import WMT20ENZHDataModule
from .wmt_roen import WMT20ROENDataModule
from .wmt_sien import WMT20SIENDataModule
from .sickr_sts import SICKRSTSDataModule
from .sts22_crosslingual_sts import STS22CrosslingualSTSDataModule
from .stsbenchmark_mteb import STSBenchmarkMTEBDataModule


def get_datamodule(
    dataset_name: str,
    model_name: str,
    max_seq_length: int,
    batch_size: int,
    *,
    tokenize_inputs: bool = True,
    combine_fields: bool = False,
    combine_separator_token: str = "[SEP]",
):
    """
    Returns an initialized LightningDataModule for the given dataset name.
    """
    name = dataset_name.lower()

    if name == "stsb":
        return STSBDataModule(
            model_name_or_path=model_name,
            max_seq_length=max_seq_length,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            tokenize_inputs=tokenize_inputs,
            combine_fields=combine_fields,
            combine_separator_token=combine_separator_token,
        )

    if name in {"stsbenchmark-mteb", "stsbenchmark", "stsbench-mteb"}:
        return STSBenchmarkMTEBDataModule(
            model_name_or_path=model_name,
            max_seq_length=max_seq_length,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            tokenize_inputs=tokenize_inputs,
            combine_fields=combine_fields,
            combine_separator_token=combine_separator_token,
        )

    if name in {"sickr-sts", "sickr", "sickr_sts"}:
        return SICKRSTSDataModule(
            model_name_or_path=model_name,
            max_seq_length=max_seq_length,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            tokenize_inputs=tokenize_inputs,
            combine_fields=combine_fields,
            combine_separator_token=combine_separator_token,
        )

    if name in {
        "sts22-crosslingual-sts",
        "sts22-xling-sts",
        "sts22",
    }:
        return STS22CrosslingualSTSDataModule(
            model_name_or_path=model_name,
            max_seq_length=max_seq_length,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            tokenize_inputs=tokenize_inputs,
            combine_fields=combine_fields,
            combine_separator_token=combine_separator_token,
        )

    if name in {"yelp", "yelp2018"}:
        return Yelp2018DataModule(
            model_name_or_path=model_name,
            max_seq_length=max_seq_length,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            tokenize_inputs=tokenize_inputs,
        )

    # ---- WMT20 MLQE Task 1 variants ----
    if name in {"wmt20-enzh", "wmt2020-enzh", "mlqe-enzh"}:
        return WMT20ENZHDataModule(
            model_name_or_path=model_name,
            max_seq_length=max_seq_length,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            tokenize_inputs=tokenize_inputs,
            combine_fields=combine_fields,
            combine_separator_token=combine_separator_token,
        )

    if name in {"wmt20-roen", "wmt2020-roen", "mlqe-roen"}:
        return WMT20ROENDataModule(
            model_name_or_path=model_name,
            max_seq_length=max_seq_length,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            tokenize_inputs=tokenize_inputs,
            combine_fields=combine_fields,
            combine_separator_token=combine_separator_token,
        )

    if name in {"wmt20-sien", "wmt2020-sien", "mlqe-sien"}:
        return WMT20SIENDataModule(
            model_name_or_path=model_name,
            max_seq_length=max_seq_length,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            tokenize_inputs=tokenize_inputs,
            combine_fields=combine_fields,
            combine_separator_token=combine_separator_token,
        )

    raise ValueError(f"Unsupported dataset name: {dataset_name}")
