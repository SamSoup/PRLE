# from .base import BasePrototypicalRegressor
# import torch
# import torch.nn as nn


# class LinearExpertPRLE(BasePrototypicalRegressor):
#     """
#     Linear experts:
#       - One nn.Linear(hidden_dim, 1) per prototype.
#       - Final prediction is soft-routed mixture of all expert outputs.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.experts = nn.ModuleList(
#             [nn.Linear(self.hidden_dim, 1) for _ in range(self.num_prototypes)]
#         )

#     def predict_from_embeddings(self, embeds: torch.Tensor) -> torch.Tensor:
#         # (B, P) soft routing over prototypes using the UNIFIED ACCESSOR
#         P = self._current_prototype_matrix()  # (P, H)
#         dists = torch.cdist(embeds.unsqueeze(1), P.unsqueeze(0)).squeeze(1)
#         sim_weights = torch.softmax(-dists, dim=1)  # (B, P)

#         # Expert outputs stacked to (B, P, 1)
#         expert_outputs = torch.stack(
#             [expert(embeds) for expert in self.experts], dim=1
#         )

#         # Weighted sum over experts -> (B, 1) -> (B,)
#         preds = (
#             (sim_weights.unsqueeze(-1) * expert_outputs).sum(dim=1).squeeze(-1)
#         )
#         return preds

# /model/linear.py
from .base import BasePrototypicalRegressor, _normalize
import torch
import torch.nn as nn


class LinearExpertPRLE(BasePrototypicalRegressor):
    """
    Linear experts:
      - One nn.Linear(hidden_dim, 1) per prototype.
      - Final prediction is soft-routed mixture of all expert outputs.
      - Routing uses the shared, normalized distance with learnable tau.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experts = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 1) for _ in range(self.num_prototypes)]
        )

    def predict_from_embeddings(self, embeds: torch.Tensor) -> torch.Tensor:
        # Distances in normalized space (matches cohesion and selection)
        dists = self.compute_distances(embeds)  # (B, P)
        weights = torch.softmax(-self.tau * dists, dim=1)  # (B, P)

        # Expert outputs: (B, P, 1)
        expert_outputs = torch.stack(
            [expert(embeds) for expert in self.experts], dim=1
        )

        # Weighted sum over experts -> (B, 1) -> (B,)
        preds = (weights.unsqueeze(-1) * expert_outputs).sum(dim=1).squeeze(-1)
        return preds
