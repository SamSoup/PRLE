from .base import BasePrototypicalRegressor
import torch
import torch.nn as nn


class MLPExpertPRLE(BasePrototypicalRegressor):
    def __init__(self, *args, mlp_layers=2, mlp_hidden=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.experts = nn.ModuleList(
            [
                self._build_mlp(self.hidden_dim, mlp_hidden, mlp_layers)
                for _ in range(self.num_prototypes)
            ]
        )

    def _build_mlp(self, input_dim, hidden_dim, layers):
        modules = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layers - 2):
            modules += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        modules += [nn.Linear(hidden_dim, 1)]
        return nn.Sequential(*modules)

    def forward(self, input_ids, attention_mask):
        embeds = self.shared_forward(input_ids, attention_mask)
        dists = torch.cdist(embeds.unsqueeze(1), self.prototypes.unsqueeze(0))
        sim_weights = torch.softmax(-dists.squeeze(1), dim=1)
        expert_outputs = torch.stack(
            [expert(embeds) for expert in self.experts], dim=1
        )
        preds = (
            (sim_weights.unsqueeze(-1) * expert_outputs).sum(dim=1).squeeze(-1)
        )
        return preds, embeds
