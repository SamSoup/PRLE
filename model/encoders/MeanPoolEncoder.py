# /model/encoder/MeanPoolEncoder.py
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def _mean_pooling(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    last_hidden_state: (B, T, H)
    attention_mask:    (B, T)
    returns:           (B, H) mean-pooled over valid tokens
    """
    # expand mask to (B, T, H)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # float
    summed = torch.sum(last_hidden_state * mask, dim=1)  # (B, H)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)  # (B, 1)
    return summed / counts


class MeanPoolEncoder(nn.Module):
    """
    HF encoder with attention-mask aware mean pooling over token embeddings.
    Expects tokenized inputs (input_ids, attention_mask, optional token_type_ids).
    Optionally L2-normalizes the pooled embedding.

    Example call inside a LightningModule:
        embeds = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        normalize: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        if gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()
        self.normalize = normalize

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns pooled sentence embeddings of shape (B, H).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
            return_dict=True,
        )
        # last_hidden_state: (B, T, H)
        last_hidden_state = outputs.last_hidden_state
        pooled = _mean_pooling(last_hidden_state, attention_mask)  # (B, H)
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled
