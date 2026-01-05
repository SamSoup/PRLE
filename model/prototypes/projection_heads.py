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
    "rbf_rff",  # random Fourier features → RBF-ish map,
    "resmlp",  # residual MLP kernel (v2 default)
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

        phi(x) = sqrt(2/D) * [cos(xW  b), sin(xW  b)]

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
        # we build W for half-dim, then concat cossin
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


class ResMLPProjectionHead(BaseProjectionHead):
    """
    Slightly higher-capacity projection head:
      - stack of Linear  nonlinearity ( dropout)
      - optional residual connection if input/output dims match
      - final LayerNorm for stability

    This is meant to mimic the kind of "kernel" head used in modern
    sentence-transformer finetuning, but still relatively cheap.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        hidden_dim = hidden_dim or out_dim

        layers = []
        last_dim = in_dim
        act_cls = nn.ReLU if activation.lower() == "relu" else nn.GELU

        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.out_proj = (
            nn.Linear(hidden_dim, out_dim)
            if hidden_dim != out_dim
            else nn.Identity()
        )
        self._out_dim = out_dim
        self.norm = nn.LayerNorm(out_dim)

        self._in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = self.out_proj(h)
        # simple residual when dimensions match
        if x.shape[-1] == h.shape[-1]:
            h = h + x
        h = self.norm(h)
        return h

    @property
    def output_dim(self) -> int:
        return self._out_dim


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

    if kind == "resmlp":
        out = output_dim or input_dim
        hidden = kwargs.get("hidden_dim", None)
        num_layers = kwargs.get("num_layers", 2)
        dropout = kwargs.get("dropout", 0.0)
        activation = kwargs.get("activation", "relu")
        return ResMLPProjectionHead(
            in_dim=input_dim,
            out_dim=out,
            hidden_dim=hidden,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
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
