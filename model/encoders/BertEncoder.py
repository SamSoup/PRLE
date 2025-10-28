from typing import Optional
import torch
from torch import nn
from transformers import AutoModel


class BertEncoder(nn.Module):
    """
    HF Transformer encoder that expects TOKENIZED inputs:
      forward(input_ids=..., attention_mask=...) -> returns CLS from last hidden state: (B, H)
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
        cls = last_hidden[:, 0, :]  # (B, H)
        return cls
