# model/base.py (revised minimal PRLE skeleton, no encoder, no cohesion/separation yet)

from typing import Optional, Dict, Any, Tuple, List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics.functional as tmf


# -------------------------------------------------
# Low-level math helpers
# -------------------------------------------------


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize last dimension."""
    return F.normalize(x, p=2, dim=-1)


def _pairwise_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: str = "cosine",
) -> torch.Tensor:
    """
    Compute distance between each row in x and each row in y.
    x: (B,H)
    y: (P,H)
    Returns: (B,P)
    """
    x_n = _normalize(x)
    y_n = _normalize(y)

    if metric == "cosine":
        # cosine distance = 1 - cosine similarity
        return 1.0 - torch.einsum("bh,ph->bp", x_n, y_n)
    elif metric == "euclidean":
        # euclidean distance on the unit sphere ~ chord distance
        return torch.cdist(x_n, y_n, p=2)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


# -------------------------------------------------
# Prototype parameter container (for 'free' mode)
# -------------------------------------------------


class PrototypeBank(nn.Module):
    """
    Learnable prototype vectors of shape (P,H) when using 'free' mode.
    """

    def __init__(self, num_prototypes: int, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_prototypes, hidden_dim))

    def forward(self) -> torch.Tensor:
        return self.weight  # (P,H)


# -------------------------------------------------
# Base LightningModule
# -------------------------------------------------


class BasePrototypicalRegressor(pl.LightningModule):
    """
    Skeleton for Prototypical Regression with Local Experts (PRLE).

    Assumptions in this revision:
    - The encoder is *not part of this module*. We never forward tokens here.
      The LightningDataModule upstream already:
        (a) computed frozen embeddings once,
        (b) serves batches shaped like:
            {
              "embeddings": FloatTensor (B,H),
              "labels":     FloatTensor (B,)
            }

    - We are currently *only* doing supervised regression loss (MSE).
      Cohesion / separation / interpretability regularizers are temporarily removed
      and will be reintroduced later in a more principled way.

    - Routing temperature / tau is removed. Routing is still hookable.

    - The model organizes functionality into conceptual stages:
        (1) Prototype definition / initialization
        (2) Expert heads
        (3) Routing / mixture weights
        (4) Aggregation into final prediction
        (5) Loss
        (6) Metrics + Lightning glue
    """

    def __init__(
        self,
        hidden_dim: int,
        num_prototypes: int,
        lr: float = 1e-4,
        output_dir: str = "outputs",
        datamodule: Optional[pl.LightningDataModule] = None,
        prototype_mode: str = "free",  # "free" | "example"
        prototype_selector: Optional[str] = None,
        seed: int = 42,
        distance: str = "cosine",  # "cosine" | "euclidean"
        mse_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule"])

        # -------------------------
        # Basic hparams
        # -------------------------
        self.hidden_dim = hidden_dim
        self.num_prototypes = int(num_prototypes)
        self.lr = lr
        self.seed = seed
        self.mse_weight = mse_weight
        self.distance_metric = distance.lower().strip()
        assert self.distance_metric in {"cosine", "euclidean"}

        # -------------------------
        # (1) Prototype state
        # -------------------------
        self.prototype_mode = prototype_mode.lower().strip()
        assert self.prototype_mode in {"free", "example"}

        # Optional hint: how to pick examples when using example mode
        self.prototype_selector = (
            None if prototype_selector is None else prototype_selector.lower()
        )

        # Storage for prototypes
        #   - free mode: learnable PrototypeBank
        #   - example mode: frozen exemplar embeddings + their indices into train set
        if self.prototype_mode == "free":
            self.prototype_bank = PrototypeBank(self.num_prototypes, hidden_dim)
            self.prototype_indices: Optional[torch.LongTensor] = (
                None  # not used
            )
            self.prototype_embeds: Optional[torch.Tensor] = None  # not used
        else:
            self.prototype_bank = None
            self.prototype_indices = None  # LongTensor (P,)
            self.prototype_embeds = None  # Tensor (P,H) on device

        # We keep a reference to the EmbeddingDataModule so we can use
        # dm.train_embeddings to initialize example-based prototypes.
        self.dm: Optional[pl.LightningDataModule] = datamodule

        # -------------------------
        # (2) Expert heads
        # -------------------------
        # Subclasses are expected to define any expert modules here.
        # e.g. a per-prototype linear regressor.
        self._init_experts_hook()

        # -------------------------
        # Bookkeeping for validation/test metrics
        # -------------------------
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []

        # -------------------------
        # I/O
        # -------------------------
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # =================================================
    # (1) Prototype Initialization / Maintenance hooks
    # =================================================

    def _init_examples_as_prototypes(
        self,
        train_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [TO IMPLEMENT IN SUBCLASS IF USING 'example' MODE]

        Goal:
            Pick which training points become prototypes and return both their
            indices and corresponding embedding vectors.

        Input:
            train_embeds: (N,H) float32 tensor from EmbeddingDataModule.train_embeddings (CPU)

        Returns:
            proto_indices: (P,) LongTensor of selected row indices
            proto_vectors: (P,H) FloatTensor of those rows' embeddings
                           (we'll move to the model device)

        The selection policy (k-means++, farthest-first, label binning, etc.)
        should live in subclass code.
        """
        raise NotImplementedError(
            "Subclass must implement _init_examples_as_prototypes() "
            "when prototype_mode='example'."
        )

    def _ensure_prototypes_ready(self) -> None:
        """
        Make sure we have a prototype matrix to route against.

        - free mode:
            prototypes are just learnable parameters in self.prototype_bank,
            so there's nothing special to do.
        - example mode:
            if we haven't yet chosen exemplar prototypes,
            call _init_examples_as_prototypes() using dm.train_embeddings.
        """
        if self.prototype_mode == "free":
            return

        if self.prototype_mode == "example" and self.prototype_embeds is None:
            if (
                self.dm is None
                or getattr(self.dm, "train_embeddings", None) is None
            ):
                raise RuntimeError(
                    "EmbeddingDataModule not attached or not set up; "
                    "cannot init example-based prototypes."
                )

            train_cpu = self.dm.train_embeddings.detach().cpu()  # (N,H)
            proto_idx, proto_vecs = self._init_examples_as_prototypes(train_cpu)

            # cache them
            self.prototype_indices = proto_idx.clone().cpu()  # (P,)
            self.prototype_embeds = proto_vecs.to(
                self.device
            ).contiguous()  # (P,H)

    def _get_prototype_matrix(self) -> torch.Tensor:
        """
        Unified accessor.
        Returns:
            (P,H) tensor of prototype embeddings on the correct device.
        """
        if self.prototype_mode == "free":
            return self.prototype_bank.weight  # (P,H) learnable parameters
        else:
            if self.prototype_embeds is None:
                raise RuntimeError("Prototypes not initialized yet.")
            return self.prototype_embeds  # (P,H) frozen snapshot

    # =================================================
    # (2) Expert Heads
    # =================================================

    def _init_experts_hook(self) -> None:
        """
        Optional constructor-time setup for expert heads.

        Example in subclass:
            self.experts = nn.ModuleList(
                [nn.Linear(self.hidden_dim, 1) for _ in range(self.num_prototypes)]
            )
        """
        return

    def _expert_outputs(
        self,
        sample_embed: torch.Tensor,  # (B,H)
    ) -> torch.Tensor:
        """
        [TO IMPLEMENT IN SUBCLASS]

        Produce each prototype expert's prediction for each sample.
        Output:
            expert_matrix: (B,P) or (B,P,1)

        - expert_matrix[b, j] is "what prototype j predicts" for sample b.
        """
        raise NotImplementedError(
            "_expert_outputs() must be implemented in subclass."
        )

    # =================================================
    # (3) Routing / Mixture Weights
    # =================================================

    def _compute_routing_distances(
        self,
        sample_embed: torch.Tensor,  # (B,H)
        proto_embed: torch.Tensor,  # (P,H)
    ) -> torch.Tensor:
        """
        Default: pure geometric distance.
        Override if you want label-aware distance, dual-space routing, etc.
        """
        return _pairwise_distance(
            sample_embed,
            proto_embed,
            metric=self.distance_metric,
        )  # (B,P)

    def _compute_routing_weights(
        self,
        distances: torch.Tensor,  # (B,P)
    ) -> torch.Tensor:
        """
        Convert distances -> mixture weights.

        We intentionally do NOT learn a temperature here yet.
        The default is just softmax(-distance).

        Override this to implement:
          - sparse gating
          - top-k masking
          - learned per-prototype radii
          - etc.
        """
        return torch.softmax(-distances, dim=1)  # (B,P)

    # =================================================
    # (4) Aggregation
    # =================================================

    def _aggregate_expert_predictions(
        self,
        expert_matrix: torch.Tensor,  # (B,P) or (B,P,1)
        weights: torch.Tensor,  # (B,P)
    ) -> torch.Tensor:
        """
        Weighted sum across prototypes → final scalar regression prediction.
        """
        if expert_matrix.dim() == 3:
            expert_matrix = expert_matrix.squeeze(-1)  # (B,P)
        preds = (weights * expert_matrix).sum(dim=1)  # (B,)
        return preds

    def forward(
        self,
        batch_embeddings: torch.Tensor,  # (B,H)
    ) -> torch.Tensor:
        """
        Full forward pass for training / inference:
            1. ensure prototypes exist
            2. compute expert outputs
            3. compute routing weights
            4. aggregate
        """
        self._ensure_prototypes_ready()

        proto_mat = self._get_prototype_matrix()  # (P,H)
        expert_matrix = self._expert_outputs(
            batch_embeddings
        )  # (B,P) or (B,P,1)
        dists = self._compute_routing_distances(
            batch_embeddings, proto_mat
        )  # (B,P)
        weights = self._compute_routing_weights(dists)  # (B,P)
        preds = self._aggregate_expert_predictions(
            expert_matrix, weights
        )  # (B,)
        return preds

    # =================================================
    # (5) Loss
    # =================================================

    def _task_loss(
        self,
        preds: torch.Tensor,  # (B,)
        labels: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        Core supervised objective (regression).
        """
        labels = labels.view_as(preds).to(preds.dtype)
        return F.mse_loss(preds, labels)

    def _compute_total_loss(
        self,
        preds: torch.Tensor,  # (B,)
        labels: torch.Tensor,  # (B,)
        embeds: torch.Tensor,  # (B,H)  (not currently used, but kept for future regularizers)
    ) -> Dict[str, torch.Tensor]:
        """
        Combine all objectives into a dict of losses to log.
        For now, it's just MSE scaled by mse_weight.
        """
        mse = self._task_loss(preds, labels)
        total = self.mse_weight * mse
        return {
            "total": total,
            "mse": mse,
        }

    # =================================================
    # (6) Metrics / Logging
    # =================================================

    def _log_epoch_metrics(
        self,
        preds_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        prefix: str,
    ) -> None:
        """
        After an epoch (val/test), compute summary stats.
        """
        if not preds_list:
            return
        preds = torch.cat(preds_list)
        labels = torch.cat(labels_list)

        mse = tmf.mean_squared_error(preds, labels)
        rho = tmf.pearson_corrcoef(preds, labels)

        self.log(f"{prefix}_mse", mse, prog_bar=True)
        self.log(f"{prefix}_corr", rho, prog_bar=True)

    # =================================================
    # Lightning-required plumbing
    # =================================================

    def _shared_step(
        self,
        batch: Dict[str, Any],
        stage: str,
    ) -> torch.Tensor:
        """
        Stage-agnostic step.
        batch is expected to have:
            batch["embeddings"]: (B,H)
            batch["labels"]:     (B,)
        """
        embeds = batch["embeddings"].to(self.device).float()  # (B,H)
        labels = batch["labels"].to(self.device).float()  # (B,)

        preds = self.forward(embeds).view(-1)  # (B,)
        losses = self._compute_total_loss(preds, labels, embeds)
        loss = losses["total"]

        # logging / buffer collection
        if stage == "train":
            self.log(
                "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True
            )
            self.log("train_loss_epoch", loss, on_step=False, on_epoch=True)
        elif stage == "val":
            self._val_preds.append(preds.detach().cpu())
            self._val_labels.append(labels.detach().cpu())
            self.log(
                "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True
            )
        elif stage == "test":
            self._test_preds.append(preds.detach().cpu())
            self._test_labels.append(labels.detach().cpu())
            self.log(
                "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, stage="test")

    def on_validation_epoch_start(self):
        self._val_preds, self._val_labels = [], []

    def on_validation_epoch_end(self):
        self._log_epoch_metrics(self._val_preds, self._val_labels, prefix="val")

    def on_test_epoch_start(self):
        self._test_preds, self._test_labels = [], []

    def on_test_epoch_end(self):
        self._log_epoch_metrics(
            self._test_preds, self._test_labels, prefix="test"
        )

    def setup(self, stage: Optional[str] = None):
        """
        Lightning hook: ensure prototypes exist before any training/val/test.
        """
        if stage in (None, "fit", "validate", "test", "predict"):
            self._ensure_prototypes_ready()

    def on_fit_start(self):
        """
        Optional debug print for sanity.
        """
        self._ensure_prototypes_ready()
        self._dump_trainable_status()

    def configure_optimizers(self):
        """
        Basic Adam over all trainable params (experts, free prototypes, etc.).
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # -------------------------------------------------
    # Debug helpers
    # -------------------------------------------------

    def _dump_trainable_status(self):
        """
        Print parameter counts for sanity.
        """

        def count_params(mod: nn.Module) -> Tuple[int, int]:
            trainable = sum(
                p.numel() for p in mod.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in mod.parameters())
            return trainable, total

        lines = []

        # prototypes
        if self.prototype_mode == "free":
            tr, tot = count_params(self.prototype_bank)
            lines.append(f"[trainable] prototypes(free): {tr:,}/{tot:,}")
        else:
            pcount = (
                0
                if self.prototype_embeds is None
                else self.prototype_embeds.numel()
            )
            lines.append(
                f"[trainable] prototypes(example): params=0 | "
                f"cached_embed_elems={pcount:,} | "
                f"indices_set={self.prototype_indices is not None}"
            )

        # experts
        expert_params = 0
        for name, mod in self.named_children():
            if name in ["prototype_bank"]:
                # skip counting prototype_bank here because we already counted it
                continue
            if isinstance(mod, nn.Module):
                expert_params += sum(
                    p.numel() for p in mod.parameters() if p.requires_grad
                )
        lines.append(f"[trainable] experts(est.): {expert_params:,}")

        total_trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        total_all = sum(p.numel() for p in self.parameters())
        lines.append(f"[trainable] TOTAL: {total_trainable:,}/{total_all:,}")

        self.print("\n".join(lines), flush=True)
