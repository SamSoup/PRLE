# model/activations.py

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def _pairwise_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: str = "cosine",
) -> torch.Tensor:
    """
    Compute retrieval-space distance between each row of x and each row of y.
    Returns (B,P).
    """
    x_n = _normalize(x)
    y_n = _normalize(y)

    if metric == "cosine":
        # cosine distance = 1 - cos_sim
        return 1.0 - torch.einsum("bh,ph->bp", x_n, y_n)
    elif metric == "euclidean":
        return torch.cdist(x_n, y_n, p=2)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


class ActivationRouter(nn.Module):
    """
    Abstract base for routing / activation.
    Subclasses must implement forward(...) returning weights (B,P).

    forward signature:
        retrieval_embeds: (B,H_retr)
        retrieval_protos: (P,H_retr)
        value_embeds:     (B,Dv)
        value_protos:     (P,Dv)

    The router may depend on retrieval space (kNN/radius/softmax over dist)
    or value space (sparse MLP gating), or both.
    """

    def __init__(self, distance_metric: str):
        super().__init__()
        distance_metric = distance_metric.lower().strip()
        assert distance_metric in {"cosine", "euclidean"}
        self.distance_metric = distance_metric

    def _distances(
        self,
        retrieval_embeds: torch.Tensor,  # (B,H)
        retrieval_protos: torch.Tensor,  # (P,H)
    ) -> torch.Tensor:
        return _pairwise_distance(
            retrieval_embeds,
            retrieval_protos,
            metric=self.distance_metric,
        )

    def enable_training(self, flag: bool) -> None:
        """
        Turn router params (if any) on/off for gradient updates.
        Default: nothing trainable, so no-op.
        Subclasses override if they have trainable params.
        """
        # default: stateless router (kNN, radius) => nothing to flip
        for p in self.parameters():
            p.requires_grad = flag
        self.train(flag)

    def forward(
        self,
        retrieval_embeds: torch.Tensor,  # (B,H)
        retrieval_protos: torch.Tensor,  # (P,H)
        value_embeds: torch.Tensor,  # (B,Dv)
        value_protos: torch.Tensor,  # (P,Dv)
    ) -> torch.Tensor:
        raise NotImplementedError

    # must return (B,P) row-normalized weights
    # each row sums to 1, weights >= 0


class KNNRouter(ActivationRouter):
    """
    Hard-ish local routing: only the top-k nearest prototypes get mass.
    Mass within that subset is softmax(-dist).
    """

    def __init__(self, distance_metric: str, k: int = 3):
        super().__init__(distance_metric=distance_metric)
        self.k = int(k)

    def forward(
        self,
        retrieval_embeds: torch.Tensor,
        retrieval_protos: torch.Tensor,
        value_embeds: torch.Tensor,
        value_protos: torch.Tensor,
    ) -> torch.Tensor:
        dists = self._distances(retrieval_embeds, retrieval_protos)  # (B,P)
        B, P = dists.shape
        k = min(self.k, P)

        knn_idx = torch.topk(dists, k, dim=1, largest=False).indices  # (B,k)

        weights = torch.zeros_like(dists)  # (B,P)

        d_knn = torch.gather(dists, 1, knn_idx)  # (B,k)
        w_knn = torch.softmax(-d_knn, dim=1)  # (B,k)

        weights.scatter_(1, knn_idx, w_knn)
        return weights  # (B,P), row sums = 1


class RadiusRouter(ActivationRouter):
    """
    Local routing by radius: any prototype within distance <= radius_threshold
    gets mass. If none qualify, the nearest one is forced active.
    Weights within the active set come from softmax(-dist).
    """

    def __init__(self, distance_metric: str, radius_threshold: float = 0.5):
        super().__init__(distance_metric=distance_metric)
        self.radius_threshold = float(radius_threshold)

    def forward(
        self,
        retrieval_embeds: torch.Tensor,
        retrieval_protos: torch.Tensor,
        value_embeds: torch.Tensor,
        value_protos: torch.Tensor,
    ) -> torch.Tensor:
        dists = self._distances(retrieval_embeds, retrieval_protos)  # (B,P)
        B, P = dists.shape

        mask = dists <= self.radius_threshold  # (B,P) bool

        # make sure each row has at least 1 active proto
        nearest_idx = torch.argmin(dists, dim=1, keepdim=True)  # (B,1)
        empty_rows = mask.sum(dim=1, keepdim=True) == 0
        mask.scatter_(1, nearest_idx, empty_rows)

        # softmax(-dist) but only over mask
        masked_scores = torch.full_like(dists, float("-inf"))
        masked_scores[mask] = -dists[mask]
        weights = torch.softmax(masked_scores, dim=1)  # (B,P)
        return weights


class SoftmaxRouter(ActivationRouter):
    """
    Dense differentiable gating with temperature annealing.

    We compute:
        τ_eff = clamp( tau_raw * tau_multiplier, tau_min, tau_max )
        weights = softmax( -τ_eff * dist )

    Where:
      - tau_raw is a learned Parameter (if training enabled).
      - tau_multiplier is a running buffer we anneal each epoch
        (e.g. multiply by 0.95 every epoch to gradually sharpen).
      - clamp(·) prevents pathological extremes.

    Why:
      * Early epochs: smoother routing → stable gradients.
      * Later epochs: sharper routing → more local experts, better ranking.
    """

    def __init__(
        self,
        distance_metric: str,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        tau_max: float = 10.0,
        anneal_gamma: float = 0.95,
    ):
        super().__init__(distance_metric=distance_metric)

        # base learnable temperature (1 scalar)
        self.tau_raw = nn.Parameter(
            torch.tensor(float(tau_init)), requires_grad=True
        )

        # multiplier that we decay each epoch (not trainable by grad)
        self.register_buffer(
            "tau_multiplier",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=True,
        )

        # hyperparams for stability
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.anneal_gamma = float(anneal_gamma)

    def enable_training(self, flag: bool) -> None:
        # Toggle gradient updates for tau_raw
        self.tau_raw.requires_grad = flag
        self.train(flag)

    @torch.no_grad()
    def epoch_anneal(self) -> None:
        """
        Called once per training epoch by LightningModule.
        We decay tau_multiplier so τ_eff gets smaller over time,
        which sharpens routing (more peaky softmax).
        """
        # multiply and clamp to avoid collapsing to 0 too fast
        new_mult = self.tau_multiplier * self.anneal_gamma
        # don't let it go negative or crazy tiny
        new_mult = torch.clamp(new_mult, min=0.05, max=100.0)
        self.tau_multiplier.copy_(new_mult)

    def _effective_tau(self) -> torch.Tensor:
        """
        Compute τ_eff = clamp(tau_raw * tau_multiplier, [tau_min, tau_max])
        Return shape (), on correct device.
        """
        tau_eff = self.tau_raw * self.tau_multiplier
        tau_eff = torch.clamp(tau_eff, self.tau_min, self.tau_max)
        return tau_eff

    def forward(
        self,
        retrieval_embeds: torch.Tensor,
        retrieval_protos: torch.Tensor,
        value_embeds: torch.Tensor,
        value_protos: torch.Tensor,
    ) -> torch.Tensor:
        dists = self._distances(retrieval_embeds, retrieval_protos)  # (B,P)
        tau_eff = self._effective_tau()  # scalar tensor
        return torch.softmax(-tau_eff * dists, dim=1)  # (B,P)


class SparseMLPRouter(ActivationRouter):
    """
    Learned sparse gating (mixture-of-experts style).

    For each (b, p), build a feature vector in VALUE space:
        concat[
            x_val[b],
            p_val[p],
            |x_val[b] - p_val[p]|,
            ||x_val[b] - p_val[p]||_2
        ]  -> shape (3*Dv + 1)

    Feed through a small MLP -> logits[b,p]
    Then use gumbel-softmax across prototypes => sparse-ish α[b,p].

    This router learns *which* prototypes should fire for a given input,
    without strictly relying on raw retrieval distance.
    """

    def __init__(
        self,
        distance_metric: str,
        proj_dim: int,
        mlp_hidden_dim: int = 64,
    ):
        super().__init__(distance_metric=distance_metric)
        feature_dim = 3 * proj_dim + 1
        self.gating_mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def enable_training(self, flag: bool) -> None:
        self.train(flag)
        for p in self.gating_mlp.parameters():
            p.requires_grad = flag

    def forward(
        self,
        retrieval_embeds: torch.Tensor,  # (B,H) (unused directly in this router)
        retrieval_protos: torch.Tensor,  # (P,H) (unused directly in this router)
        value_embeds: torch.Tensor,  # (B,Dv)
        value_protos: torch.Tensor,  # (P,Dv)
    ) -> torch.Tensor:
        B, Dv = value_embeds.shape
        P, Dp = value_protos.shape
        assert Dv == Dp, "SparseMLPRouter: value-space dim mismatch"

        # pairwise features
        x_exp = value_embeds.unsqueeze(1).expand(B, P, Dv)  # (B,P,Dv)
        p_exp = value_protos.unsqueeze(0).expand(B, P, Dv)  # (B,P,Dv)
        diff_abs = torch.abs(x_exp - p_exp)  # (B,P,Dv)
        dist_l2 = torch.norm(diff_abs, p=2, dim=2, keepdim=True)  # (B,P,1)

        feat = torch.cat(
            [x_exp, p_exp, diff_abs, dist_l2], dim=2
        )  # (B,P,3*Dv+1)

        logits = self.gating_mlp(feat).squeeze(-1)  # (B,P)

        # sparse routing via gumbel softmax
        weights = F.gumbel_softmax(
            logits,
            tau=1.0,
            hard=False,
            dim=1,
        )  # (B,P)
        return weights


def build_activation_router(
    strategy: str,
    distance_metric: str,
    knn_k: int = 3,
    radius_threshold: float = 0.5,
    proj_dim: Optional[int] = None,
    mlp_hidden_dim: int = 64,
) -> ActivationRouter:
    """
    Factory for building activation / routing modules.
    """
    strategy = strategy.lower().strip()
    if strategy == "knn":
        return KNNRouter(distance_metric=distance_metric, k=knn_k)
    if strategy == "radius":
        return RadiusRouter(
            distance_metric=distance_metric, radius_threshold=radius_threshold
        )
    if strategy == "softmax":
        return SoftmaxRouter(distance_metric=distance_metric)
    if strategy == "mlp_sparse":
        assert proj_dim is not None, "SparseMLPRouter requires proj_dim"
        return SparseMLPRouter(
            distance_metric=distance_metric,
            proj_dim=proj_dim,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    # default: plain softmax-style dense routing
    return SoftmaxRouter(distance_metric=distance_metric)
