from .base import BasePrototypicalRegressor
import torch


class NaiveExpertPRLE(BasePrototypicalRegressor):
    """
    Naive experts:
      - Each prototype maps to a fixed scalar bin center in [0, 1].
      - Prediction is the softmax-weighted average of bin centers using similarity to prototypes.
      - Gradients flow through distances (prototypes + encoder if trainable).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register as buffer so it moves with .to(device) and is saved in state_dict
        self.register_buffer(
            "expert_bins_edges",
            torch.linspace(0, 1, steps=self.num_prototypes + 1),
        )

    def predict_from_embeddings(self, embeds: torch.Tensor) -> torch.Tensor:
        # bin centers from edges
        bin_centers = (
            self.expert_bins_edges[:-1] + self.expert_bins_edges[1:]
        ) / 2.0  # (P,)
        # distances & soft routing
        dists = torch.cdist(
            embeds.unsqueeze(1), self.prototype_bank.weight.unsqueeze(0)
        ).squeeze(
            1
        )  # (B, P)
        sim_weights = torch.softmax(-dists, dim=1)  # (B, P)
        preds = (sim_weights * bin_centers.unsqueeze(0)).sum(dim=1)  # (B,)
        return preds
