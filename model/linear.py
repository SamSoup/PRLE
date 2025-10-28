# model/linear.py

from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR  # <-- new import

from model.base import BasePrototypicalRegressor


ExpertInitStrategy = Literal[
    "random",
    "meta_linear_regression",
    "meta_classification_head",
    "meta_svr",
]


class LinearExpertPRLE(BasePrototypicalRegressor):
    """
    PRLE instantiation where each prototype has its own linear regressor
    (a 1-layer nn.Linear) over the VALUE-space embedding.

    During inference:
      - Base class handles routing/activation and produces weights (B,P)
      - Here we produce expert outputs (B,P,1)
      - Base class combines via weighted sum

    Expert initialization strategies:
      1. "random":
            Leave per-prototype Linear layers randomly initialized.

      2. "meta_linear_regression":
            Fit a *single* global LinearRegression (sklearn) from
            (VALUE-space embedding) -> label on the training set.
            Copy that W,b to all experts.

      3. "meta_classification_head":
            Train a tiny 1-layer torch linear head with MSE on the full
            training set for a few steps (full-batch).
            Copy those learned params to all experts.

      4. "meta_svr":
            Fit a single global Support Vector Regressor (SVR) with a linear
            kernel on (VALUE-space embedding) -> label.
            Copy that W,b (coef_, intercept_) to all experts.

            Notes:
            - We use kernel="linear" so coef_ exists and the model is equivalent
              to a linear regressor with margin-based regularization.
            - This can give a slightly more robust global slope than plain OLS.
    """

    def __init__(
        self,
        *args,
        expert_init_strategy: ExpertInitStrategy = "random",
        **kwargs,
    ):
        # Store strategy so _init_experts_hook() can read it after super().__init__
        self.expert_init_strategy = expert_init_strategy
        super().__init__(*args, **kwargs)

    # -------------------------------------------------
    # Expert construction / initialization
    # -------------------------------------------------

    def _init_experts_hook(self) -> None:
        """
        Build self.experts and meta-initialize them based on expert_init_strategy.
        """
        proj_dim = self.prototype_manager.proj_dim
        P = self.num_prototypes

        # one linear regressor per prototype
        self.experts = nn.ModuleList([nn.Linear(proj_dim, 1) for _ in range(P)])

        # If random init: nothing else to do.
        if self.expert_init_strategy == "random":
            return

        # We need training data (cached embeddings + labels) for meta init.
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

        # Pull out training embeddings/labels
        with torch.no_grad():
            train_emb = (
                self.dm.train_embeddings.detach().cpu()
            )  # (N,H_retrieval)
            train_y = self.dm.train_labels.detach().cpu().float()  # (N,)

            # Project to VALUE space using the current projection head
            train_val = self.prototype_manager.project_embeddings(
                train_emb
            ).cpu()  # (N,proj_dim)

        # -------------------------------
        # Strategy: meta_linear_regression
        # -------------------------------
        if self.expert_init_strategy == "meta_linear_regression":
            print(
                "[experts:init] meta_linear_regression: fitting sklearn LinearRegression"
            )
            X = train_val.numpy()  # (N,Dv)
            y = train_y.numpy()  # (N,)

            lr = LinearRegression()
            lr.fit(X, y)

            # sklearn LinearRegression has shape (Dv,) coef_ and scalar intercept_
            W = torch.from_numpy(lr.coef_.reshape(1, -1)).float()  # (1,Dv)
            b = torch.tensor(lr.intercept_, dtype=torch.float32).view(1)  # (1,)

            for expert in self.experts:
                expert.weight.data.copy_(W)
                expert.bias.data.copy_(b)

            print(
                "[experts:init] meta_linear_regression: copied global W,b to all experts"
            )
            return

        # -------------------------------
        # Strategy: meta_classification_head
        # (really just a learned regression head via torch)
        # -------------------------------
        if self.expert_init_strategy == "meta_classification_head":
            print(
                "[experts:init] meta_classification_head: training torch Linear head on full train set"
            )

            head = nn.Linear(proj_dim, 1)
            opt = torch.optim.Adam(head.parameters(), lr=1e-2)

            X = train_val  # (N,Dv) torch on CPU
            y = train_y.view(-1, 1)  # (N,1)

            head.train()
            n_steps = 200
            for step in range(n_steps):
                opt.zero_grad()
                pred = head(X)  # (N,1)
                loss = F.mse_loss(pred, y)
                loss.backward()
                opt.step()

                if (step + 1) % 50 == 0:
                    print(
                        f"[experts:init] warm-start step {step+1}/{n_steps}, loss={loss.item():.4f}"
                    )

            with torch.no_grad():
                W = head.weight.data.clone()  # (1,Dv)
                B = head.bias.data.clone()  # (1,)

            for expert in self.experts:
                expert.weight.data.copy_(W)
                expert.bias.data.copy_(B)

            print(
                "[experts:init] meta_classification_head: broadcast warm-start params"
            )
            return

        # -------------------------------
        # Strategy: meta_svr
        # -------------------------------
        if self.expert_init_strategy == "meta_svr":
            print(
                "[experts:init] meta_svr: fitting sklearn SVR(kernel='linear')"
            )

            X = train_val.numpy()  # (N,Dv)
            y = train_y.numpy()  # (N,)

            # We'll use a linear SVR so that coef_ / intercept_ exist and define a linear model
            svr = SVR(kernel="linear", C=1.0, epsilon=0.1, verbose=True)
            svr.fit(X, y)

            # For linear-kernel SVR:
            #   svr.coef_ -> shape (1, Dv)
            #   svr.intercept_ -> shape (1,)
            W = torch.from_numpy(svr.coef_.reshape(1, -1)).float()  # (1,Dv)
            b = torch.from_numpy(svr.intercept_.reshape(1)).float()  # (1,)

            for expert in self.experts:
                expert.weight.data.copy_(W)
                expert.bias.data.copy_(b)

            print(
                "[experts:init] meta_svr: broadcasted SVR-derived W,b to all experts"
            )
            return

        # -------------------------------
        # Fallback / unknown strategy
        # -------------------------------
        print(
            f"[experts:init] Unknown expert_init_strategy={self.expert_init_strategy}, "
            "leaving random init."
        )

    # -------------------------------------------------
    # Expert forward
    # -------------------------------------------------

    def _expert_outputs(
        self,
        value_embeds: Tensor,  # (B, proj_dim)
    ) -> Tensor:
        """
        Apply each prototype's linear regressor to each sample.
        Return shape (B,P,1).
        """
        outs = []
        for expert in self.experts:
            outs.append(expert(value_embeds))  # (B,1)
        return torch.stack(outs, dim=1)  # (B,P,1)

    # -------------------------------------------------
    # Expert trainability control
    # -------------------------------------------------

    def _enable_expert_training(self, flag: bool) -> None:
        """
        Turn gradient updates for expert heads on/off.
        Called by the base class during EM-phase flips.
        """
        self.experts.train(flag)
        for expert in self.experts:
            for p in expert.parameters():
                p.requires_grad = flag
