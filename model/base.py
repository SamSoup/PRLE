from typing import Optional, Dict, Any, Tuple, List
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics.functional as tmf
import torchmetrics

from model.prototypes import PrototypeManager, build_prototype_manager
from model.activations import build_activation_router


class BasePrototypicalRegressor(pl.LightningModule):
    """
    PRLE base class with two regimes:

      - v1 (use_v2_loss == False):
          * Prototypes initialized / reasoned about in retrieval space.
          * Router primarily uses retrieval-space distances
            (with optional value-space info for MLP router).
          * Multi-term loss:
                task + metric + proto_self + local + consistency
          * Optional EM-style geometry vs experts alternation.

      - v2 (use_v2_loss == True):
          * Value space is pre-trained by a separate kernel stage.
          * Prototypes are selected in VALUE space (k-medoids / kmeans++ etc.),
            but still correspond to concrete training examples.
          * Router operates purely in VALUE space (KNN/softmax/radius/MLP).
          * Loss is MoE-style:
                task_main + expert_fit + anchor_route + balance + entropy
          * Geometry is frozen; only experts + router train.

    Subclasses define the expert family by implementing:
        _init_experts_hook
        _enable_expert_training
        _expert_outputs
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
        projection_type: str = "linear",
        projection_kwargs: Optional[Dict[str, Any]] = None,
        em_alt_training: bool = False,  # v1-only: alternate per epoch
        # activation / gating config
        gating_strategy: str = "softmax",  # "knn" | "radius" | "softmax" | "mlp_sparse"
        knn_k: int = 3,
        radius_threshold: float = 0.5,
        activation_mlp_hidden_dim: int = 64,
        expert_mlp_hidden_dim: int = 64,
        # loss weighting per phase (v1)
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
        # -------- PRLE v2 loss config --------
        use_v2_loss: bool = False,
        lambda_task_main: float = 1.0,
        lambda_expert_fit: float = 0.5,
        lambda_anchor_route: float = 0.5,
        lambda_balance: float = 0.1,
        lambda_entropy: float = 0.0,
    ):
        super().__init__()
        self.expert_mlp_hidden_dim = int(expert_mlp_hidden_dim)
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
            projection_type=projection_type,
            projection_kwargs=projection_kwargs,
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
            mlp_hidden_dim=activation_mlp_hidden_dim,
        )

        # -------------------
        # experts (subclass builds self.experts etc.)
        # -------------------
        self._init_experts_hook()

        # -------------------
        # EM alternation config (v1)
        # -------------------
        self.em_alt_training = bool(em_alt_training)
        # possible stages: "geometry", "experts", "all_unfrozen"
        self._training_stage = (
            "geometry" if self.em_alt_training else "all_unfrozen"
        )

        # -------------------
        # Phase-weighted loss lambdas (v1)
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
        # v2 loss config
        # -------------------
        self.use_v2_loss = bool(use_v2_loss)
        self.lambda_task_main = float(lambda_task_main)
        self.lambda_expert_fit = float(lambda_expert_fit)
        self.lambda_anchor_route = float(lambda_anchor_route)
        self.lambda_balance = float(lambda_balance)
        self.lambda_entropy = float(lambda_entropy)

        # -------------------
        # buffers for eval logging
        # -------------------
        self._val_preds: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []

        self.output_dir = output_dir
        if self.output_dir is not None:
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
    # Forward internals (v1 vs v2)
    # ------------------------------------------------------------

    def _forward_internals_v1(
        self,
        retrieval_embeds: torch.Tensor,  # (B,H_retrieval)
    ) -> Dict[str, torch.Tensor]:
        """
        v1 pipeline:
          - Prototypes in retrieval space.
          - Router distances in retrieval space.
          - Experts consume value-space embeddings.
        """
        batch_device = retrieval_embeds.device

        # Prototypes (retrieval + value)
        retrieval_protos = self.prototype_manager.get_retrieval_prototypes(
            device=batch_device
        )  # (P,H)
        value_embeds = self.prototype_manager.project_embeddings(
            retrieval_embeds
        )  # (B,Dv)
        value_protos = self.prototype_manager.get_value_prototypes(
            device=batch_device
        )  # (P,Dv)

        # Router in retrieval space (legacy behavior for KNN/softmax/radius)
        weights = self.activation_router(
            retrieval_embeds=retrieval_embeds,
            retrieval_protos=retrieval_protos,
            value_embeds=value_embeds,
            value_protos=value_protos,
        )  # (B,P)

        # Experts run in value space
        expert_matrix = self._expert_outputs(value_embeds)  # (B,P) or (B,P,1)

        preds = self._aggregate_expert_predictions(expert_matrix, weights)

        return {
            "retrieval_protos": retrieval_protos,
            "value_protos": value_protos,
            "value_embeds": value_embeds,
            "weights": weights,
            "expert_matrix": expert_matrix,
            "preds": preds,
        }

    def _forward_internals_v2(
        self,
        retrieval_embeds: torch.Tensor,  # (B,H_retrieval)
    ) -> Dict[str, torch.Tensor]:
        """
        v2 pipeline:
          - Value space is the primary geometry.
          - Prototypes are chosen in value space (at init time).
          - Router operates entirely in value space:
                distances(value_embeds, value_protos)
          - Experts consume value-space embeddings.
        """
        batch_device = retrieval_embeds.device

        # Still expose retrieval_protos (for interpretability/debugging),
        # but routing itself ignores them.
        retrieval_protos = self.prototype_manager.get_retrieval_prototypes(
            device=batch_device
        )  # (P,H)

        # Project batch + prototypes to value space
        value_embeds = self.prototype_manager.project_embeddings(
            retrieval_embeds
        )  # (B,Dv)
        value_protos = self.prototype_manager.get_value_prototypes(
            device=batch_device
        )  # (P,Dv)

        # Router DISPATCH: pass value-space tensors also as "retrieval_*"
        # so that KNN/Softmax/Radius routers now operate in value space.
        weights = self.activation_router(
            retrieval_embeds=value_embeds,  # <- value space
            retrieval_protos=value_protos,  # <- value space
            value_embeds=value_embeds,
            value_protos=value_protos,
        )  # (B,P)

        expert_matrix = self._expert_outputs(value_embeds)  # (B,P) or (B,P,1)
        preds = self._aggregate_expert_predictions(expert_matrix, weights)

        return {
            "retrieval_protos": retrieval_protos,
            "value_protos": value_protos,
            "value_embeds": value_embeds,
            "weights": weights,
            "expert_matrix": expert_matrix,
            "preds": preds,
        }

    def _forward_internals(
        self,
        retrieval_embeds: torch.Tensor,  # (B,H_retrieval)
    ) -> Dict[str, torch.Tensor]:
        """
        Dispatch to v1 or v2 pipeline depending on use_v2_loss.
        """
        self._ensure_prototypes_ready()
        if self.use_v2_loss:
            return self._forward_internals_v2(retrieval_embeds)
        else:
            return self._forward_internals_v1(retrieval_embeds)

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
    # Loss terms (v1 + v2)
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

        Used in v1 loss; v2 folds this behavior into expert-fit + anchor routing.
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
        """
        Responsibility-weighted per-expert MSE.
        Used in both v1 (as 'local loss') and v2 (as 'expert_fit').
        """
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
        """
        v1: discourage assigning large weight to prototypes far away in value space.
        In v2 we typically drop this (kernel handles geometry).
        """
        B, Dv = value_embeds.shape
        P, Dp = value_protos.shape
        assert Dv == Dp, "consistency_loss: dim mismatch"

        vb = value_embeds.unsqueeze(1).expand(B, P, Dv)  # (B,P,Dv)
        vp = value_protos.unsqueeze(0).expand(B, P, Dv)  # (B,P,Dv)

        dist_sq = torch.sum((vb - vp) ** 2, dim=2)  # (B,P)
        weighted = weights * dist_sq  # (B,P)
        return weighted.mean()

    # -------- v2-specific helper losses --------

    def _anchor_routing_loss(
        self,
        weights: torch.Tensor,  # (B,P)
        batch_indices: Optional[torch.Tensor],  # (B,) or None
    ) -> torch.Tensor:
        """
        Encourage router to send each prototype's anchor example to that prototype.

        For anchor example a(p) of prototype p, we want w[a(p), p] to be high.
        Only anchors present in the current batch contribute.
        """
        if batch_indices is None:
            return torch.tensor(0.0, device=weights.device)

        pm = self.prototype_manager
        if (
            not hasattr(pm, "prototype_indices")
            or pm.prototype_indices is None
            or pm.prototype_indices.numel() == 0
        ):
            return torch.tensor(0.0, device=weights.device)

        B, P = weights.shape
        batch_indices = batch_indices.view(B).to(pm.prototype_indices.device)
        proto_indices = pm.prototype_indices.view(1, P)  # (1,P)
        batch_expanded = batch_indices.view(B, 1)  # (B,1)

        # mask[b,p] = True iff example b is the anchor for prototype p
        mask = batch_expanded.eq(proto_indices)  # (B,P)
        if not mask.any():
            return torch.tensor(0.0, device=weights.device)

        anchor_weights = weights[mask]  # (num_anchors_in_batch,)
        eps = 1e-8
        return -torch.log(anchor_weights + eps).mean()

    def _load_balance_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Mixture-of-experts style load-balancing:
        encourage all experts to receive similar average responsibility.
        """
        B, P = weights.shape
        if B == 0 or P == 0:
            return torch.tensor(0.0, device=weights.device)
        usage = weights.mean(dim=0)  # (P,)
        return P * (usage.pow(2).sum())

    def _routing_entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Optional routing-entropy term.

        - With lambda_entropy > 0, minimizing this loss tends to REDUCE entropy
          (sharper routing).
        - With lambda_entropy < 0, minimizing this loss tends to INCREASE entropy
          (smoother routing).
        """
        B, P = weights.shape
        if B == 0 or P == 0:
            return torch.tensor(0.0, device=weights.device)
        eps = 1e-8
        w = torch.clamp(weights, eps, 1.0)
        entropy = -(w * w.log()).sum(dim=1).mean()  # average H over batch
        return entropy

    # ------------------------------------------------------------
    # Phase-aware weighting (v1)
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
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute scalar loss and its components.

        If self.use_v2_loss is True, use PRLE v2 objectives:
          - L_task_main
          - L_expert_fit (local responsibility)
          - L_anchor_route
          - L_balance
          - L_entropy (optional)

        Otherwise, fall back to the original v1 multi-term loss.
        """
        preds = internals["preds"]  # (B,)
        expert_matrix = internals["expert_matrix"]  # (B,P)/(B,P,1)
        weights = internals["weights"]  # (B,P)
        value_embeds = internals["value_embeds"]  # (B,Dv)
        value_protos = internals["value_protos"]  # (P,Dv)

        if self.use_v2_loss:
            l_task = self._task_loss(preds, labels)
            l_fit = self._local_loss(expert_matrix, weights, labels)
            l_anchor = self._anchor_routing_loss(weights, batch_indices)
            l_balance = self._load_balance_loss(weights)
            l_entropy = self._routing_entropy_loss(weights)

            total = (
                self.lambda_task_main * l_task
                + self.lambda_expert_fit * l_fit
                + self.lambda_anchor_route * l_anchor
                + self.lambda_balance * l_balance
                + self.lambda_entropy * l_entropy
            )

            return {
                "total": total,
                "task": l_task.detach(),
                "expert_fit": l_fit.detach(),
                "anchor_route": l_anchor.detach(),
                "balance": l_balance.detach(),
                "entropy": l_entropy.detach(),
            }

        # ---- original v1 behavior ----
        c = self._phase_coeffs()

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
        """
        Aggregate predictions/labels over an epoch and log metrics.

        Metrics on the RAW label scale:
          - mse, rmse
          - pearson, spearman, kendall

        If a label_max is available (self.label_max or dm.label_max), we also
        compute the same metrics on [0,1]-normalized scores.
        """
        if not preds_list:
            return

        preds = torch.cat(preds_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)

        # ---- RAW SCALE METRICS ----
        mse_raw = tmf.mean_squared_error(preds, labels)
        rmse_raw = torch.sqrt(mse_raw)
        pearson_raw = tmf.pearson_corrcoef(preds, labels)
        spearman_raw = tmf.spearman_corrcoef(preds, labels)

        # Kendall's tau via the metric class (not functional)
        kendall_metric = torchmetrics.KendallRankCorrCoef().to(self.device)
        kendall_raw = kendall_metric(preds, labels)

        self.log(f"{prefix}_mse", mse_raw, prog_bar=True)
        self.log(f"{prefix}_rmse", rmse_raw, prog_bar=False)
        self.log(f"{prefix}_pearson", pearson_raw, prog_bar=True)
        self.log(f"{prefix}_spearman", spearman_raw, prog_bar=False)
        self.log(f"{prefix}_kendall", kendall_raw, prog_bar=False)

        # ---- NORMALIZED METRICS (OPTIONAL) ----
        label_max = None
        if hasattr(self, "label_max") and self.label_max is not None:
            label_max = float(self.label_max)
        elif self.dm is not None and hasattr(self.dm, "label_max"):
            label_max = float(self.dm.label_max)

        if label_max is not None and label_max > 0:
            preds_norm = preds / label_max
            labels_norm = labels / label_max

            mse_norm = tmf.mean_squared_error(preds_norm, labels_norm)
            rmse_norm = torch.sqrt(mse_norm)
            pearson_norm = tmf.pearson_corrcoef(preds_norm, labels_norm)
            spearman_norm = tmf.spearman_corrcoef(preds_norm, labels_norm)

            kendall_norm_metric = torchmetrics.KendallRankCorrCoef().to(
                self.device
            )
            kendall_norm = kendall_norm_metric(preds_norm, labels_norm)

            self.log(f"{prefix}_mse_norm", mse_norm, prog_bar=False)
            self.log(f"{prefix}_rmse_norm", rmse_norm, prog_bar=False)
            self.log(f"{prefix}_pearson_norm", pearson_norm, prog_bar=False)
            self.log(f"{prefix}_spearman_norm", spearman_norm, prog_bar=False)
            self.log(f"{prefix}_kendall_norm", kendall_norm, prog_bar=False)

    # ------------------------------------------------------------
    # Lightning plumbing
    # ------------------------------------------------------------

    def _init_prototypes_v2_value_space(self) -> None:
        """
        v2 prototype init:
          - Compute ALL train embeddings in VALUE space via the kernel
            projection head.
          - Run the same clustering / seeding strategy as v1, but on
            value-space embeddings (k-medoids, kmeans++, random_real, ...).
          - Use the resulting indices to select retrieval-space prototypes,
            so each prototype still corresponds to a concrete training example.
        """
        pm = self.prototype_manager

        # Already initialized (from checkpoint or prior call)
        if getattr(pm, "_is_initialized_buf", None) is not None:
            if int(pm._is_initialized_buf.item()) == 1:
                return
        if getattr(pm, "_is_initialized", False):
            return

        if (
            self.dm is None
            or getattr(self.dm, "train_embeddings", None) is None
        ):
            raise RuntimeError(
                "EmbeddingDataModule not attached or not setup; cannot init prototypes (v2)."
            )

        train_retr = self.dm.train_embeddings.detach().cpu()  # (N,H)
        N = train_retr.size(0)
        if N == 0:
            raise RuntimeError(
                "No train embeddings available for prototype init."
            )

        # Compute value-space embeddings for all train points in manageable chunks
        device = self.device
        pm_device = device  # keep projection head on model device
        pm.to(pm_device)

        chunks: List[torch.Tensor] = []
        chunk_size = 4096
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = train_retr[start:end].to(pm_device)
            with torch.no_grad():
                val_chunk = self.prototype_manager.project_embeddings(chunk)
            chunks.append(val_chunk.detach().cpu())
        train_val = torch.cat(chunks, dim=0)  # (N,Dv)

        # Reuse the same seeding strategies but in VALUE space:
        # we only need the indices, not the proto vectors they return.
        proto_idx, _ = pm._init_raw_prototypes(train_val)  # type: ignore[attr-defined]

        # Map back to retrieval space for actual stored prototype vectors
        proto_idx = proto_idx.long().detach().cpu()
        proto_vecs_retr = train_retr[proto_idx]  # (P,H)

        pm._register_retrieval_prototypes(proto_vecs_retr)  # type: ignore[attr-defined]
        pm.prototype_indices.data.copy_(proto_idx.long())
        pm._is_initialized = True
        if hasattr(pm, "_is_initialized_buf"):
            pm._is_initialized_buf.data.copy_(
                torch.tensor(1, dtype=torch.uint8)
            )

    def _ensure_prototypes_ready(self) -> None:
        """
        Make sure prototypes are initialized exactly once.

        v1:
          - call PrototypeManager.initialize_from_train_embeddings(...)
            using retrieval-space embeddings.

        v2:
          - run _init_prototypes_v2_value_space(), which picks anchors in
            value space but stores retrieval-space vectors.
        """
        pm = self.prototype_manager

        # If already initialized (fresh run or restored from checkpoint), stop.
        if getattr(pm, "_is_initialized_buf", None) is not None:
            if int(pm._is_initialized_buf.item()) == 1:
                return

        if self.use_v2_loss:
            self._init_prototypes_v2_value_space()
            return

        # v1: use existing retrieval-space initializer
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
        batch_indices = batch.get("indices", None)
        if batch_indices is not None:
            batch_indices = batch_indices.to(self.device)

        internals = self._forward_internals(retrieval_embeds)
        losses = self._compute_total_loss(
            retrieval_embeds, labels, internals, batch_indices=batch_indices
        )
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
    # EM-style alternation per epoch (v1) + v2 override
    # ------------------------------------------------------------

    def on_train_epoch_start(self):
        """
        If use_v2_loss=True, we run in experts-only mode:
          - geometry (projection + prototypes) frozen
          - router + experts train

        Else if em_alt_training=True (v1):
          - alternate between geometry vs experts phases.

        Else:
          - all components train ("all_unfrozen").
        """
        # router may implement epoch-wise annealing (e.g. temperature)
        if hasattr(self.activation_router, "epoch_anneal"):
            self.activation_router.epoch_anneal()

        # ---- v2 override: always experts-only, geometry frozen ----
        if self.use_v2_loss:
            self._training_stage = "experts"
            # geometry OFF
            self.prototype_manager.enable_projection_training(False)
            self.prototype_manager.enable_prototype_training(False)
            # experts + router ON
            self._enable_expert_training(True)
            self._enable_router_training(True)
            return

        # ---- original v1 behavior ----
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
        In v2 we typically keep prototypes fixed (no snapping), so we skip.
        """
        if self.use_v2_loss:
            return

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
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=5e-7
        )

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
