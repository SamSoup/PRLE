# model/kernel.py

from __future__ import annotations
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model.prototypes.projection_heads import build_projection_head


class KernelMetricLearner(pl.LightningModule):
    """
    Stage 1: learn a value-space kernel g(·) with metric learning only.

    Inputs:
      - embeddings: (B, H) frozen retrieval embeddings
      - labels:     (B,)

    Objective:
      - pairwise similarity regression (label gap ↔ cosine similarity)
      - optional triplet ranking
      - uniformity / spread (SimCLR-style)
    """

    def __init__(
        self,
        hidden_dim: int,
        proj_dim: Optional[int] = None,
        projection_type: str = "resmlp",
        projection_kwargs: Optional[Dict[str, Any]] = None,
        label_max: float = 1.0,
        lambda_pair: float = 1.0,
        lambda_triplet: float = 0.0,
        lambda_uniform: float = 0.01,
        lr: float = 2e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.label_max = label_max
        self.lambda_pair = lambda_pair
        self.lambda_triplet = lambda_triplet
        self.lambda_uniform = lambda_uniform
        self.lr = lr

        proj_dim_eff = proj_dim if proj_dim is not None else hidden_dim

        self.projection_head = build_projection_head(
            kind=projection_type,
            input_dim=hidden_dim,
            output_dim=proj_dim_eff,
            **(projection_kwargs or {}),
        )

    # -------------------------
    # core forward
    # -------------------------

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        z = self.projection_head(embeddings)
        z = F.normalize(z, p=2, dim=-1)
        return z

    # -------------------------
    # losses
    # -------------------------

    def _pairwise_similarity_regression_loss(
        self,
        z: torch.Tensor,  # (B,D)
        labels: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        B = labels.size(0)
        if B < 2:
            return torch.zeros((), device=z.device)

        z = F.normalize(z, p=2, dim=-1)
        sim = z @ z.T  # (B,B) cosine-like similarity

        y = labels.view(B, 1)
        diff = (y - y.T).abs()  # (B,B)

        max_diff = (
            self.label_max
            if getattr(self, "label_max", None) is not None
            else diff.max().detach().clamp(min=1e-6)
        )
        target = 1.0 - diff / max_diff

        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        mask = ~eye

        if not mask.any():
            return torch.zeros((), device=z.device)

        return F.mse_loss(sim[mask], target[mask])

    def _triplet_ranking_loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.1,
        pos_threshold: float = 0.1,
        neg_threshold: float = 0.3,
    ) -> torch.Tensor:
        if self.lambda_triplet <= 0.0:
            return torch.zeros((), device=z.device)

        B = labels.size(0)
        if B < 3:
            return torch.zeros((), device=z.device)

        z = F.normalize(z, p=2, dim=-1)
        sim = z @ z.T  # (B,B)
        y = labels.view(B, 1)
        diff = (y - y.T).abs()

        loss_terms = []

        for i in range(B):
            pos_mask = (diff[i] <= pos_threshold) & (
                torch.arange(B, device=z.device) != i
            )
            neg_mask = (diff[i] >= neg_threshold) & (
                torch.arange(B, device=z.device) != i
            )

            if not pos_mask.any() or not neg_mask.any():
                continue

            pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1)
            neg_idx = torch.nonzero(neg_mask, as_tuple=False).view(-1)

            pos_j = pos_idx[torch.randint(len(pos_idx), (1,))[0]]
            neg_k = neg_idx[torch.randint(len(neg_idx), (1,))[0]]

            s_pos = sim[i, pos_j]
            s_neg = sim[i, neg_k]
            loss_terms.append(F.relu(margin - s_pos + s_neg))

        if not loss_terms:
            return torch.zeros((), device=z.device)

        return torch.stack(loss_terms).mean()

    def _uniformity_loss(self, z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
        B = z.size(0)
        if B < 2:
            return torch.zeros((), device=z.device)

        z = F.normalize(z, p=2, dim=-1)
        pdist = torch.pdist(z, p=2)  # pairwise L2
        return torch.log(torch.exp(-t * pdist.pow(2)).mean() + 1e-8)

    # -------------------------
    # Lightning hooks
    # -------------------------

    def training_step(self, batch, batch_idx):
        x = batch["embeddings"].float()
        labels = batch["labels"].float()
        z = self.forward(x)

        L_pair = self._pairwise_similarity_regression_loss(z, labels)
        L_triplet = self._triplet_ranking_loss(z, labels)
        L_uniform = self._uniformity_loss(z)

        loss = (
            self.lambda_pair * L_pair
            + self.lambda_triplet * L_triplet
            + self.lambda_uniform * L_uniform
        )

        self.log(
            "kernel/train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "kernel/train_pair_loss",
            L_pair.detach(),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "kernel/train_triplet_loss",
            L_triplet.detach(),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "kernel/train_uniform_loss",
            L_uniform.detach(),
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["embeddings"].float()
        labels = batch["labels"].float()
        z = self.forward(x)
        L_pair = self._pairwise_similarity_regression_loss(z, labels)
        self.log(
            "kernel/val_pair_loss",
            L_pair,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return L_pair

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
