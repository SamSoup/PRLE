# model/base.py

from typing import Optional, Dict, Any, Tuple, List
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics.functional as tmf

from model.prototypes import PrototypeManager, build_prototype_manager
from model.activations import build_activation_router


class BasePrototypicalRegressor(pl.LightningModule):
    """
    PRLE base class.

    Pieces:
      - PrototypeManager: prototypes + projection head + metric learning.
      - ActivationRouter: routing / gating logic.
      - Subclass (LinearExpertPRLE, etc.): per-prototype experts.

    Responsibilities:
      - forward() composition
      - multi-term loss (task, metric, proto_self, local, consistency)
      - EM-style alternation control
      - logging / metrics
    """

    def __init__(
        self,
        hidden_dim: int,  # retrieval_dim
        num_prototypes: int,
        lr: float = 1e-4,
        output_dir: str = "outputs",
        datamodule: Optional[pl.LightningDataModule] = None,
        # prototype / geometry config
        distance: str = "cosine",
        seed: int = 42,
        init_strategy: str = "random_real",
        map_strategy: str = "none",
        trainable_prototypes: bool = False,
        # value space / metric learning config
        use_value_space: bool = True,
        proj_dim: Optional[int] = None,
        em_alt_training: bool = False,  # alternate per epoch between geometry vs experts
        # activation / gating config
        gating_strategy: str = "softmax",  # "knn" | "radius" | "softmax" | "mlp_sparse"
        knn_k: int = 3,
        radius_threshold: float = 0.5,
        mlp_hidden_dim: int = 64,
        # loss weighting per phase
        lambda_task_geom: float = 1.0,
        lambda_task_expert: float = 1.0,
        lambda_metric_geom: float = 1.0,
        lambda_metric_expert: float = 0.0,
        lambda_proto_self_geom: float = 0.0,
        lambda_proto_self_expert: float = 1.0,
        lambda_local_geom: float = 0.0,
        lambda_local_expert: float = 1.0,
        lambda_consistency_geom: float = 1.0,
        lambda_consistency_expert: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule"])

        # -------------------
        # core config
        # -------------------
        self.hidden_dim = hidden_dim
        self.num_prototypes = int(num_prototypes)
        self.lr = lr
        self.seed = seed

        self.distance_metric = distance.lower().strip()
        assert self.distance_metric in {"cosine", "euclidean"}

        # datamodule with cached embeddings
        self.dm: Optional[pl.LightningDataModule] = datamodule

        # -------------------
        # prototype manager (geometry)
        # -------------------
        self.prototype_manager: PrototypeManager = build_prototype_manager(
            num_prototypes=self.num_prototypes,
            retrieval_dim=self.hidden_dim,
            proj_dim=proj_dim,
            init_strategy=init_strategy,
            map_strategy=map_strategy,
            distance_metric=self.distance_metric,
            trainable_prototypes=trainable_prototypes,
            use_value_space=use_value_space,
            seed=self.seed,
        )

        # -------------------
        # activation router (routing/selection of prototypes)
        # -------------------
        self.activation_router = build_activation_router(
            strategy=gating_strategy,
            distance_metric=self.distance_metric,
            knn_k=knn_k,
            radius_threshold=radius_threshold,
            proj_dim=self.prototype_manager.proj_dim,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        # -------------------
        # experts (subclass builds self.experts etc.)
        # -------------------
        self._init_experts_hook()

        # -------------------
        # EM alternation config
        # -------------------
        self.em_alt_training = bool(em_alt_training)
        # possible stages: "geometry", "experts", "all_unfrozen"
        self._training_stage = (
            "geometry" if self.em_alt_training else "all_unfrozen"
        )

        # -------------------
        # Phase-weighted loss lambdas
        # -------------------
        self.lambda_task_geom = lambda_task_geom
        self.lambda_task_expert = lambda_task_expert

        self.lambda_metric_geom = lambda_metric_geom
        self.lambda_metric_expert = lambda_metric_expert

        self.lambda_proto_self_geom = lambda_proto_self_geom
        self.lambda_proto_self_expert = lambda_proto_self_expert

        self.lambda_local_geom = lambda_local_geom
        self.lambda_local_expert = lambda_local_expert

        self.lambda_consistency_geom = lambda_consistency_geom
        self.lambda_consistency_expert = lambda_consistency_expert

        # -------------------
        # buffers for eval logging
        # -------------------
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Hooks subclasses must implement
    # ------------------------------------------------------------

    def _init_experts_hook(self) -> None:
        return

    def _enable_expert_training(self, flag: bool) -> None:
        raise NotImplementedError

    def _expert_outputs(
        self,
        value_embeds: torch.Tensor,  # (B,Dv)
    ) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------
    # Helper: enable/disable router training in EM
    # ------------------------------------------------------------

    def _enable_router_training(self, flag: bool) -> None:
        """
        Delegate to activation_router.enable_training(flag)
        so that in geometry-phase we can freeze routing/gating params,
        and in experts-phase we can unfreeze them.
        """
        self.activation_router.enable_training(flag)

    # ------------------------------------------------------------
    # Forward internals
    # ------------------------------------------------------------

    def _forward_internals(
        self,
        retrieval_embeds: torch.Tensor,  # (B,H_retrieval)
    ) -> Dict[str, torch.Tensor]:
        """
        Produce predictions + intermediates for losses.

        Returns:
            retrieval_protos: (P,H)
            value_protos:     (P,Dv)
            value_embeds:     (B,Dv)
            weights:          (B,P)
            expert_matrix:    (B,P) or (B,P,1)
            preds:            (B,)
        """
        self._ensure_prototypes_ready()

        batch_device = retrieval_embeds.device

        # Get prototypes on SAME DEVICE as the batch.
        retrieval_protos = self.prototype_manager.get_retrieval_prototypes(
            device=batch_device
        )  # (P,H)

        # Project embeddings to value space
        value_embeds = self.prototype_manager.project_embeddings(
            retrieval_embeds
        )  # (B,Dv)

        # Project prototype embeddings to value space
        value_protos = self.prototype_manager.get_value_prototypes(
            device=batch_device
        )  # (P,Dv)

        # Router/gating: weights over prototypes
        weights = self.activation_router(
            retrieval_embeds=retrieval_embeds,
            retrieval_protos=retrieval_protos,
            value_embeds=value_embeds,
            value_protos=value_protos,
        )  # (B,P)

        # Experts: per-proto prediction for each example
        expert_matrix = self._expert_outputs(value_embeds)  # (B,P) or (B,P,1)

        # Final scalar prediction
        preds = self._aggregate_expert_predictions(
            expert_matrix, weights
        )  # (B,)

        return {
            "retrieval_protos": retrieval_protos,
            "value_protos": value_protos,
            "value_embeds": value_embeds,
            "weights": weights,
            "expert_matrix": expert_matrix,
            "preds": preds,
        }

    def forward(
        self,
        retrieval_embeds: torch.Tensor,
    ) -> torch.Tensor:
        outs = self._forward_internals(retrieval_embeds)
        return outs["preds"]

    def _aggregate_expert_predictions(
        self,
        expert_matrix: torch.Tensor,  # (B,P) or (B,P,1)
        weights: torch.Tensor,  # (B,P)
    ) -> torch.Tensor:
        if expert_matrix.dim() == 3:
            expert_matrix = expert_matrix.squeeze(-1)
        preds = (weights * expert_matrix).sum(dim=1)  # (B,)
        return preds

    # ------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------

    def _task_loss(
        self,
        preds: torch.Tensor,  # (B,)
        labels: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        labels = labels.view_as(preds).to(preds.dtype)
        return F.mse_loss(preds, labels)

    def _metric_learning_loss(
        self,
        retrieval_embeds: torch.Tensor,  # (B,H)
        labels: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        value_embeds = self.prototype_manager.project_embeddings(
            retrieval_embeds
        )
        return self.prototype_manager.metric_learning_loss(
            value_embeds=value_embeds,
            labels=labels,
        )

    def _proto_self_loss(self) -> torch.Tensor:
        """
        Prototype self-fit:
          each prototype's expert should predict its own anchor's label.
        """
        if self.dm is None:
            return torch.tensor(0.0, device=self.device)

        pm = self.prototype_manager

        # Need stored prototype_indices
        if (
            not hasattr(pm, "prototype_indices")
            or pm.prototype_indices is None
            or pm.prototype_indices.numel() == 0
        ):
            return torch.tensor(0.0, device=self.device)

        # train_embeddings is on CPU → indices must be CPU
        anchor_idx_cpu = pm.prototype_indices.detach().cpu().long()  # (P,)

        if (
            not hasattr(self.dm, "train_embeddings")
            or self.dm.train_embeddings is None
            or not hasattr(self.dm, "train_labels")
            or self.dm.train_labels is None
        ):
            return torch.tensor(0.0, device=self.device)

        # Gather anchors on CPU then move to model device
        retr_anchor = (
            self.dm.train_embeddings[anchor_idx_cpu].to(self.device).float()
        )  # (P,H)
        y_anchor = (
            self.dm.train_labels[anchor_idx_cpu].to(self.device).float()
        )  # (P,)

        # project anchors into value space
        val_anchor = self.prototype_manager.project_embeddings(
            retr_anchor
        )  # (P,Dv)

        # run all experts on those anchors
        expert_matrix = self._expert_outputs(val_anchor)  # (P,P) or (P,P,1)
        if expert_matrix.dim() == 3:
            expert_matrix = expert_matrix.squeeze(-1)

        # diagonal = expert p on its own prototype p
        diag_preds = torch.diagonal(expert_matrix, dim1=0, dim2=1)  # (P,)
        y_anchor = y_anchor.to(diag_preds.dtype)

        return F.mse_loss(diag_preds, y_anchor)

    def _local_loss(
        self,
        expert_matrix: torch.Tensor,  # (B,P)/(B,P,1)
        weights: torch.Tensor,  # (B,P)
        labels: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        if expert_matrix.dim() == 3:
            expert_matrix = expert_matrix.squeeze(-1)
        B, P = expert_matrix.shape
        labels_col = labels.view(B, 1).to(expert_matrix.dtype)  # (B,1)
        err_sq = (expert_matrix - labels_col) ** 2  # (B,P)
        weighted_err = weights * err_sq  # (B,P)
        return weighted_err.mean()

    def _consistency_loss(
        self,
        value_embeds: torch.Tensor,  # (B,Dv)
        value_protos: torch.Tensor,  # (P,Dv)
        weights: torch.Tensor,  # (B,P)
    ) -> torch.Tensor:
        B, Dv = value_embeds.shape
        P, Dp = value_protos.shape
        assert Dv == Dp, "consistency_loss: dim mismatch"

        vb = value_embeds.unsqueeze(1).expand(B, P, Dv)  # (B,P,Dv)
        vp = value_protos.unsqueeze(0).expand(B, P, Dv)  # (B,P,Dv)

        dist_sq = torch.sum((vb - vp) ** 2, dim=2)  # (B,P)
        weighted = weights * dist_sq  # (B,P)
        return weighted.mean()

    # ------------------------------------------------------------
    # Phase-aware weighting
    # ------------------------------------------------------------

    def _phase_coeffs(self) -> Dict[str, float]:
        if self._training_stage == "geometry":
            return {
                "task": self.lambda_task_geom,
                "metric": self.lambda_metric_geom,
                "proto_self": self.lambda_proto_self_geom,
                "local": self.lambda_local_geom,
                "consistency": self.lambda_consistency_geom,
            }
        elif self._training_stage == "experts":
            return {
                "task": self.lambda_task_expert,
                "metric": self.lambda_metric_expert,
                "proto_self": self.lambda_proto_self_expert,
                "local": self.lambda_local_expert,
                "consistency": self.lambda_consistency_expert,
            }
        else:  # "all_unfrozen"
            return {
                "task": 0.5 * (self.lambda_task_geom + self.lambda_task_expert),
                "metric": 0.5
                * (self.lambda_metric_geom + self.lambda_metric_expert),
                "proto_self": 0.5
                * (self.lambda_proto_self_geom + self.lambda_proto_self_expert),
                "local": 0.5
                * (self.lambda_local_geom + self.lambda_local_expert),
                "consistency": 0.5
                * (
                    self.lambda_consistency_geom
                    + self.lambda_consistency_expert
                ),
            }

    def _compute_total_loss(
        self,
        retrieval_embeds: torch.Tensor,  # (B,H)
        labels: torch.Tensor,  # (B,)
        internals: Dict[str, torch.Tensor],  # forward intermediates
    ) -> Dict[str, torch.Tensor]:
        c = self._phase_coeffs()

        preds = internals["preds"]  # (B,)
        expert_matrix = internals["expert_matrix"]  # (B,P)/(B,P,1)
        weights = internals["weights"]  # (B,P)
        value_embeds = internals["value_embeds"]  # (B,Dv)
        value_protos = internals["value_protos"]  # (P,Dv)

        l_task = self._task_loss(preds, labels)
        l_metric = self._metric_learning_loss(retrieval_embeds, labels)
        l_pself = self._proto_self_loss()
        l_local = self._local_loss(expert_matrix, weights, labels)
        l_cons = self._consistency_loss(value_embeds, value_protos, weights)

        total = (
            c["task"] * l_task
            + c["metric"] * l_metric
            + c["proto_self"] * l_pself
            + c["local"] * l_local
            + c["consistency"] * l_cons
        )

        return {
            "total": total,
            "task": l_task.detach(),
            "metric": l_metric.detach(),
            "proto_self": l_pself.detach(),
            "local": l_local.detach(),
            "consistency": l_cons.detach(),
        }

    # ------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------

    def _log_epoch_metrics(
        self,
        preds_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        prefix: str,
    ) -> None:
        if not preds_list:
            return
        preds = torch.cat(preds_list)
        labels = torch.cat(labels_list)
        mse = tmf.mean_squared_error(preds, labels)
        rho = tmf.pearson_corrcoef(preds, labels)
        self.log(f"{prefix}_mse", mse, prog_bar=True)
        self.log(f"{prefix}_corr", rho, prog_bar=True)

    # ------------------------------------------------------------
    # Lightning plumbing
    # ------------------------------------------------------------

    def _ensure_prototypes_ready(self) -> None:
        """
        Make sure prototypes are initialized exactly once.

        We now rely on PrototypeManager._is_initialized_buf, which is saved
        in checkpoints and starts at 0. After we populate real prototypes
        from training embeddings, we flip it to 1.
        """
        pm = self.prototype_manager

        # If already initialized (fresh run or restored from checkpoint), stop.
        if getattr(pm, "_is_initialized_buf", None) is not None:
            if int(pm._is_initialized_buf.item()) == 1:
                return

        # Otherwise we are in a fresh run (no checkpoint init yet)
        # and we need to initialize from train data.
        if (
            self.dm is None
            or getattr(self.dm, "train_embeddings", None) is None
        ):
            raise RuntimeError(
                "EmbeddingDataModule not attached or not setup; cannot init prototypes."
            )

        train_cpu = self.dm.train_embeddings.detach().cpu()
        pm.initialize_from_train_embeddings(train_cpu)

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        retrieval_embeds = batch["embeddings"].to(self.device).float()  # (B,H)
        labels = batch["labels"].to(self.device).float()  # (B,)

        internals = self._forward_internals(retrieval_embeds)
        losses = self._compute_total_loss(retrieval_embeds, labels, internals)
        loss = losses["total"]

        if stage == "train":
            self.log(
                "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True
            )
            self.log("train_loss_epoch", loss, on_step=False, on_epoch=True)
        elif stage == "val":
            preds = internals["preds"].detach().cpu()
            self._val_preds.append(preds)
            self._val_labels.append(labels.detach().cpu())
            self.log(
                "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True
            )
        else:  # "test"
            preds = internals["preds"].detach().cpu()
            self._test_preds.append(preds)
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

    # ------------------------------------------------------------
    # EM-style alternation per epoch
    # ------------------------------------------------------------

    def on_train_epoch_start(self):
        """
        If em_alt_training=True, flip between:
          - geometry phase: train projection/prototypes, freeze experts+router
          - experts phase:  freeze geometry, train experts+router

        If em_alt_training=False:
          everything is trainable ("all_unfrozen").
        """
        # NEW: gently anneal router temperature each epoch if it supports it
        if hasattr(self.activation_router, "epoch_anneal"):
            self.activation_router.epoch_anneal()

        if not self.em_alt_training:
            self._training_stage = "all_unfrozen"
            # everybody ON
            self.prototype_manager.enable_projection_training(True)
            self.prototype_manager.enable_prototype_training(True)
            self._enable_expert_training(True)
            self._enable_router_training(True)
            return

        # alternate stage each epoch
        if self._training_stage == "geometry":
            self._training_stage = "experts"
        else:
            self._training_stage = "geometry"

        if self._training_stage == "geometry":
            # geometry ON
            self.prototype_manager.enable_projection_training(True)
            self.prototype_manager.enable_prototype_training(True)
            # experts OFF
            self._enable_expert_training(False)
            self._enable_router_training(False)

        elif self._training_stage == "experts":
            # geometry OFF
            self.prototype_manager.enable_projection_training(False)
            self.prototype_manager.enable_prototype_training(False)
            # experts ON
            self._enable_expert_training(True)
            self._enable_router_training(True)

    def on_train_epoch_end(self):
        """
        Optional snapping (keeps prototypes tied to real examples).
        """
        if (
            self.dm is not None
            and hasattr(self.dm, "train_embeddings")
            and self.dm.train_embeddings is not None
        ):
            self.prototype_manager.periodic_snap(
                self.dm.train_embeddings.detach().cpu()
            )

    # ------------------------------------------------------------
    # setup / optim / debug
    # ------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        # Proactively ensure prototypes exist so logs/param-counts
        # are stable before training begins.
        if stage in (None, "fit", "validate", "test", "predict"):
            self._ensure_prototypes_ready()

    def on_fit_start(self):
        self._ensure_prototypes_ready()
        self._dump_trainable_status()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _dump_trainable_status(self):
        """
        Print who's trainable right now. Just for sanity / debug prints.
        """

        def count_params(mod: nn.Module) -> Tuple[int, int]:
            tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            tot = sum(p.numel() for p in mod.parameters())
            return tr, tot

        lines = []

        # PrototypeManager
        pm_tr, pm_tot = count_params(self.prototype_manager)
        lines.append(f"[trainable] prototype_manager: {pm_tr:,}/{pm_tot:,}")

        # ActivationRouter
        ar_tr, ar_tot = count_params(self.activation_router)
        lines.append(f"[trainable] activation_router: {ar_tr:,}/{ar_tot:,}")

        # Experts (rough est.)
        expert_params_trainable = 0
        expert_params_total = 0
        for name, module in self.named_children():
            if name in ["prototype_manager", "activation_router"]:
                continue
            if isinstance(module, nn.Module):
                for p in module.parameters():
                    expert_params_total += p.numel()
                    if p.requires_grad:
                        expert_params_trainable += p.numel()
        lines.append(
            f"[trainable] experts(est.): {expert_params_trainable:,}/{expert_params_total:,}"
        )

        total_trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        total_all = sum(p.numel() for p in self.parameters())
        lines.append(f"[trainable] TOTAL: {total_trainable:,}/{total_all:,}")

        self.print("\n".join(lines), flush=True)
