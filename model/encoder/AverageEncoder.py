from typing import Optional
import torch
from torch import nn
from transformers import AutoModel


class AverageEncoder(nn.Module):
    """
    HF Transformer encoder that expects TOKENIZED inputs:
      forward(input_ids=..., attention_mask=...) -> mean-pool over tokens: (B, H)
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        if attention_mask is None:
            # simple average across sequence length
            return last_hidden.mean(dim=1)

        # mask-aware mean
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (last_hidden * mask).sum(dim=1)  # (B, H)
        denom = mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
        return summed / denom
