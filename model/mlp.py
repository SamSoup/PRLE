# model/mlp.py

from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm
from model.base import BasePrototypicalRegressor


# How we initialize the per-prototype MLP experts
MLPExpertInitStrategy = Literal[
    "random",
    "meta_fullbatch",  # train one global MLP on full train set, copy weights to all experts
]


class MLPExpertPRLE(BasePrototypicalRegressor):
    """
    PRLE variant where each prototype owns a small MLP expert.

    Per-prototype expert architecture:
        Linear(proj_dim -> H)
        ReLU
        Linear(H -> 1)

    H = expert_mlp_hidden_dim (provided via BasePrototypicalRegressor.__init__)

    expert_init_strategy:
      - "random":
            leave each expert randomly initialized.

      - "meta_fullbatch":
            1) Train ONE shared MLP of the same shape on the entire training set
               (value-space embeddings -> labels) with full-batch GD.
            2) Copy that trained MLP's weights into every per-prototype expert.
            This gives each local expert a decent global prior instead of pure random.

    All routing, prototype geometry, EM-phase freezing, etc. is handled by the base.
    """

    def __init__(
        self,
        *args,
        expert_init_strategy: MLPExpertInitStrategy = "random",
        **kwargs,
    ):
        # We must set this BEFORE calling super().__init__(),
        # because BasePrototypicalRegressor.__init__() will call
        # self._init_experts_hook() as part of construction.
        self.expert_init_strategy = expert_init_strategy

        # super().__init__ will:
        #   - build PrototypeManager / ActivationRouter
        #   - store self.expert_mlp_hidden_dim
        #   - call self._init_experts_hook() (which depends on
        #     self.expert_init_strategy and self.expert_mlp_hidden_dim)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers to build / warm-start experts
    # ------------------------------------------------------------------

    def _build_single_expert(self, in_dim: int) -> nn.Module:
        """
        Create one prototype expert MLP:
            proj_dim -> hidden -> ReLU -> 1
        """
        hidden_dim = int(self.expert_mlp_hidden_dim)
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @torch.no_grad()
    def _copy_weights_expert(
        self,
        src: nn.Module,
        dst: nn.Module,
    ) -> None:
        """
        Copy parameters layer-by-layer from src expert MLP to dst.
        Assumes architecture:
          [0] = Linear(in_dim -> hidden)
          [1] = ReLU
          [2] = Linear(hidden -> 1)
        """
        # first Linear
        dst[0].weight.data.copy_(src[0].weight.data)
        dst[0].bias.data.copy_(src[0].bias.data)
        # ReLU has no params
        # final Linear
        dst[2].weight.data.copy_(src[2].weight.data)
        dst[2].bias.data.copy_(src[2].bias.data)

    def _train_meta_fullbatch_head(
        self,
        train_val: Tensor,  # (N, proj_dim) value-space embeddings
        train_y: Tensor,  # (N,) float labels
        proj_dim: int,
        steps: int = 200,
        lr: float = 5e-5,
    ) -> nn.Module:
        """
        Train a single shared MLP on the *entire* training set (full batch),
        so that it roughly learns the global regression mapping.
        We'll then clone its weights into each expert as a smart init.

        Returns:
            trained nn.Sequential([...]) with [Linear, ReLU, Linear]
        """
        head = self._build_single_expert(proj_dim)
        opt = torch.optim.Adam(head.parameters(), lr=lr)

        X = train_val  # (N, proj_dim) torch (CPU)
        y = train_y.view(-1, 1)  # (N,1)

        head.train()
        for step in tqdm(range(steps)):
            opt.zero_grad()
            pred = head(X)  # (N,1)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()

            if (step + 1) % 50 == 0:
                print(
                    f"[experts:init] meta_fullbatch step {step+1}/{steps}, "
                    f"loss={loss.item():.4f}"
                )

        return head.eval()

    # ------------------------------------------------------------------
    # Expert construction / initialization
    # ------------------------------------------------------------------

    def _init_experts_hook(self) -> None:
        """
        Build self.experts (ModuleList of per-prototype MLPs)
        and optionally warm-start them via meta_fullbatch.
        This gets called inside BasePrototypicalRegressor.__init__().
        """
        proj_dim = self.prototype_manager.proj_dim
        P = self.num_prototypes

        # 1) Build one MLP expert per prototype
        self.experts = nn.ModuleList(
            [self._build_single_expert(proj_dim) for _ in range(P)]
        )

        # 2) If we're just doing random init, nothing else to do
        if self.expert_init_strategy == "random":
            return

        # 3) We only know how to warm-start if we can see the dm's cached train data
        if (
            self.dm is None
            or getattr(self.dm, "train_embeddings", None) is None
            or getattr(self.dm, "train_labels", None) is None
        ):
            print(
                "[experts:init] No datamodule embeddings/labels available; "
                f"falling back to random init for {self.expert_init_strategy}"
            )
            return

        # Prepare full training set in VALUE space
        with torch.no_grad():
            train_emb = (
                self.dm.train_embeddings.detach().cpu()
            )  # (N, H_retrieval)
            train_y = self.dm.train_labels.detach().cpu().float()  # (N,)

            train_val = self.prototype_manager.project_embeddings(
                train_emb
            ).cpu()  # (N, proj_dim)

        if self.expert_init_strategy == "meta_fullbatch":
            print(
                "[experts:init] meta_fullbatch: training shared MLP on full train set"
            )
            meta_head = self._train_meta_fullbatch_head(
                train_val=train_val,
                train_y=train_y,
                proj_dim=proj_dim,
                steps=200,
                lr=1e-2,
            )

            # broadcast learned weights into each prototype expert
            for expert in self.experts:
                self._copy_weights_expert(meta_head, expert)

            print(
                "[experts:init] meta_fullbatch: broadcast warm-start params to all experts"
            )

        else:
            # Unknown strategy: leave as random
            print(
                f"[experts:init] Unknown expert_init_strategy="
                f"{self.expert_init_strategy}, leaving random init."
            )

    # ------------------------------------------------------------------
    # Forward: apply expert heads
    # ------------------------------------------------------------------

    def _expert_outputs(
        self,
        value_embeds: Tensor,  # (B, proj_dim)
    ) -> Tensor:
        """
        Run each prototype's expert MLP on each batch element.

        value_embeds: (B, Dv) in VALUE space (after projection head)
        returns: (B, P, 1)
        """
        outs = []
        for expert in self.experts:
            outs.append(expert(value_embeds))  # (B,1)
        return torch.stack(outs, dim=1)  # (B,P,1)

    # ------------------------------------------------------------------
    # Trainability toggle for EM alternation
    # ------------------------------------------------------------------

    def _enable_expert_training(self, flag: bool) -> None:
        """
        Flip requires_grad and .train() status for all expert MLPs.
        Called by BasePrototypicalRegressor.on_train_epoch_start()
        when it switches between "geometry" and "experts" phases.
        """
        self.experts.train(flag)
        for expert in self.experts:
            for p in expert.parameters():
                p.requires_grad = flag
