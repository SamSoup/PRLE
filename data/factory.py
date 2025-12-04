# data/factory.py

from .stsb import STSBDataModule
from .yelp import Yelp2018DataModule
from .sickr_sts import SICKRSTSDataModule
from .sts22_crosslingual_sts import STS22CrosslingualSTSDataModule
from .stsbenchmark_mteb import STSBenchmarkMTEBDataModule
from .wmt_en_zh import WMT20ENZHDataModule
from .wmt_en_ru import WMT20RUENDataModule
from .wmt_si_en import WMT20SIENDataModule


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
    if name in {"wmt20-enzh", "wmt20-en-zh", "wmt20-zhen", "wmt20-zh-en"}:
        return WMT20ENZHDataModule(
            model_name_or_path=model_name,
            max_seq_length=max_seq_length,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            tokenize_inputs=tokenize_inputs,
            combine_fields=combine_fields,
            combine_separator_token=combine_separator_token,
        )

    if name in {"wmt20-ruen", "wmt20-ru-en", "wmt20-enru", "wmt20-en-ru"}:
        return WMT20RUENDataModule(
            model_name_or_path=model_name,
            max_seq_length=max_seq_length,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            tokenize_inputs=tokenize_inputs,
            combine_fields=combine_fields,
            combine_separator_token=combine_separator_token,
        )

    if name in {
        "wmt20-sien",
        "wmt20-si-en",
        "wmt20-ensi",
        "wmt20-en-en",
    }:
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
