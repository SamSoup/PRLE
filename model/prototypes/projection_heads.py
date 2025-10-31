# model/prototypes/projection_heads.py

from __future__ import annotations
from typing import Optional, Literal, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


ProjectionKind = Literal[
    "linear",  # plain nn.Linear (your current behavior)
    "mlp",  # small MLP, still learnable
    "rbf_rff",  # random Fourier features → RBF-ish map
]


class BaseProjectionHead(nn.Module):
    """
    Common interface so PrototypeManager can stay simple.
    Must provide:
      - forward(x) -> (B, Dv)
      - output_dim: int property
      - enable_training(flag)
    """

    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    def enable_training(self, flag: bool) -> None:
        self.train(flag)
        for p in self.parameters():
            p.requires_grad = flag


class LinearProjectionHead(BaseProjectionHead):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    @property
    def output_dim(self) -> int:
        return self.proj.out_features


class MLPProjectionHead(BaseProjectionHead):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_dim, out_dim)
        layers = []
        d_in = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.GELU())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, out_dim))
        self.net = nn.Sequential(*layers)
        self._out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def output_dim(self) -> int:
        return self._out_dim


class RBFRandomFourierProjectionHead(BaseProjectionHead):
    """
    Approximates an RBF kernel feature map using Random Fourier Features:

        phi(x) = sqrt(2/D) * [cos(xW + b), sin(xW + b)]

    where:
      - W ~ N(0, 1/sigma^2)
      - b ~ Uniform(0, 2π)

    D here is *half* the final dimension; we concatenate cos and sin.

    This is a good “kernel trick in value space” that is still
    fully compatible with your per-prototype linear experts.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        sigma: float = 1.0,
        learnable_rff: bool = False,
    ):
        """
        Args:
            in_dim: retrieval/value input dim
            out_dim: final feature dim AFTER cos/sin concat
            sigma: RBF lengthscale; smaller → sharper kernel
            learnable_rff: if True, W and b are nn.Parameters
        """
        super().__init__()
        # we build W for half-dim, then concat cos+sin
        half_dim = out_dim // 2
        self.in_dim = in_dim
        self.out_dim_final = 2 * half_dim

        # sample W ~ N(0, 1/sigma^2)
        W = torch.randn(in_dim, half_dim) / sigma
        b = 2 * math.pi * torch.rand(half_dim)

        if learnable_rff:
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer("W", W)
            self.register_buffer("b", b)

        self.learnable_rff = learnable_rff
        self.sigma = sigma
        self.scaling = math.sqrt(2.0 / float(self.out_dim_final))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        proj = x @ self.W  # (B, half_dim)
        proj = proj + self.b  # broadcast
        cos_feat = torch.cos(proj)
        sin_feat = torch.sin(proj)
        feats = torch.cat([cos_feat, sin_feat], dim=-1)  # (B, 2*half_dim)
        return self.scaling * feats

    @property
    def output_dim(self) -> int:
        return self.out_dim_final

    def enable_training(self, flag: bool) -> None:
        # we only enable grads on W/b if they are parameters
        if not self.learnable_rff:
            # keep buffers frozen
            return
        super().enable_training(flag)


def build_projection_head(
    kind: ProjectionKind,
    input_dim: int,
    output_dim: Optional[int],
    **kwargs: Any,
) -> BaseProjectionHead:
    """
    Factory for projection heads.
    """
    if kind == "linear":
        out = output_dim or input_dim
        return LinearProjectionHead(input_dim, out, bias=False)

    if kind == "mlp":
        out = output_dim or input_dim
        hidden = kwargs.get("hidden_dim", None)
        num_layers = kwargs.get("num_layers", 2)
        act = kwargs.get("activation", "gelu")
        return MLPProjectionHead(
            input_dim,
            out,
            hidden_dim=hidden,
            num_layers=num_layers,
            activation=act,
        )

    if kind in {"rbf", "rbf_rff"}:
        out = output_dim or 256
        sigma = kwargs.get("sigma", 1.0)
        learnable_rff = kwargs.get("learnable_rff", False)
        return RBFRandomFourierProjectionHead(
            in_dim=input_dim,
            out_dim=out,
            sigma=sigma,
            learnable_rff=learnable_rff,
        )

    raise ValueError(f"Unknown projection head kind: {kind}")
