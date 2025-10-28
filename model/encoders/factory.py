from typing import Literal, Any

from .MeanPoolEncoder import MeanPoolEncoder
from .SentenceEncoder import SentenceEncoder
from .BertEncoder import BertEncoder
from .AverageEncoder import AverageEncoder

EncoderType = Literal["sentence", "bert", "average"]


def get_encoder(
    encoder_type: EncoderType,
    *,
    model_name: str,
    **kwargs: Any,
):
    """
    Factory for encoders.

    Args:
      encoder_type: "sentence" | "bert" | "average"
      model_name:   HF model id, e.g. "sentence-transformers/all-MiniLM-L6-v2" or "roberta-base"
      kwargs:
        - For SentenceEncoder:
            cache_dir: str
            normalize_embeddings: bool
            batch_size: int
        - For BertEncoder / AverageEncoder:
            (no extra required)

    Returns:
      nn.Module with forward() matching BasePrototypicalRegressor expectations.
    """
    key = str(encoder_type).lower()
    if key == "sentence":
        return SentenceEncoder(
            model_name=model_name,
            **kwargs,
        )
    elif key == "bert":
        return BertEncoder(model_name=model_name)
    elif key == "average":
        return AverageEncoder(model_name=model_name)
    elif key == "mean":
        return MeanPoolEncoder(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported encoder_type: {encoder_type}")
