# /model/base.py
# Prototypical regression w/ optional exemplar prototypes.
# - One-time exemplar selection at trainer startup (setup(stage))
# - Cache ONLY selected training examples (raw or tokenized)
# - Step-wise refresh of exemplar embeddings if encoder trains end-to-end
# - Persist + restore prototypes & examples in Lightning checkpoints
# - No silhouette/auto-K, no periodic reselection, no val-MSE gating

from typing import Optional, List, Dict, Tuple, Any, Union
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics.functional as tmf

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN  # assume availability
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm  # assume installed


# ------------------ small helpers ------------------
def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def _cdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, p=2)


def _farthest_first(X: torch.Tensor, k: int) -> torch.LongTensor:
    """k-center greedy on cosine distance. X on CPU, shape (N,H), normalized inside."""
    N = X.size(0)
    if k >= N:
        return torch.arange(N, dtype=torch.long)
    Xn = _normalize(X)
    idx0 = torch.randint(0, N, (1,)).item()
    chosen = [idx0]
    dmin = 1 - (Xn @ Xn[idx0])  # cosine distance
    for _ in range(1, k):
        j = torch.argmax(dmin).item()
        chosen.append(j)
        dmin = torch.minimum(dmin, 1 - (Xn @ Xn[j]))
    return torch.tensor(chosen, dtype=torch.long)


def _nearest_point_indices(
    embeds: torch.Tensor, centers: np.ndarray, k: int
) -> torch.LongTensor:
    """Map each center to the nearest training example index (greedy, no dup)."""
    E = embeds  # (N,H) CPU torch
    C = torch.from_numpy(centers).to(E.dtype)
    D = torch.cdist(E, C, p=2)  # (N,k)
    used = set()
    out = []
    for j in range(C.size(0)):
        col = D[:, j]
        _, order = torch.sort(col)
        pick = None
        for idx in order:
            ii = int(idx.item())
            if ii not in used:
                pick = ii
                used.add(ii)
                break
        out.append(pick if pick is not None else int(order[0].item()))
    return torch.tensor(out, dtype=torch.long)


def _cluster_means_to_indices(
    embeds: torch.Tensor, labels: np.ndarray
) -> torch.LongTensor:
    """For label array (>=1 non-noise cluster), compute per-cluster mean and map to nearest exemplar index."""
    E = embeds.numpy()
    ids = []
    for lab in sorted(set(labels)):
        if lab == -1:  # ignore noise (DBSCAN/HDBSCAN)
            continue
        mask = labels == lab
        if mask.sum() <= 0:
            continue
        mean = E[mask].mean(axis=0, keepdims=True)  # (1,H)
        idx = _nearest_point_indices(embeds, mean, 1).item()
        ids.append(idx)
    if len(ids) == 0:
        # fallback: pick a single medoid-like point (global mean)
        ids = [
            _nearest_point_indices(
                embeds, E.mean(axis=0, keepdims=True), 1
            ).item()
        ]
    return torch.tensor(ids, dtype=torch.long)


# ------------------ free prototype bank ------------------
class PrototypeBank(nn.Module):
    def __init__(self, num_prototypes: int, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_prototypes, hidden_dim))

    def forward(self):
        return self.weight


# ------------------ main module ------------------
class BasePrototypicalRegressor(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        num_prototypes: int,
        mse_weight: float = 1.0,
        cohesion_weight: float = 0.0,
        separation_weight: float = 0.0,
        lr: float = 1e-4,
        output_dir: str = "outputs",
        freeze_encoder: bool = False,
        # Exemplar options
        datamodule=None,  # pass the DM instance
        prototype_mode: str = "free",  # "free" | "example"
        prototype_selector: Optional[
            str
        ] = None,  # "kmedoids" | "kmeans++" | "farthest" | "dbscan" | "hdbscan" | None
        seed: int = 42,
        # Routing options
        distance: str = "cosine",  # "cosine" | "euclidean"
        tau_init: float = 1.0,  # learnable temperature for routing
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "datamodule"])

        # encoder
        self.encoder = encoder
        if isinstance(self.encoder, nn.Module):
            for p in self.encoder.parameters():
                p.requires_grad = not freeze_encoder
            self.encoder.train(not freeze_encoder)

        # core hparams
        self.hidden_dim = hidden_dim
        self.num_prototypes = int(num_prototypes)
        self.lr = lr
        self.mse_weight = mse_weight
        self.cohesion_weight = cohesion_weight
        self.separation_weight = separation_weight
        self.output_dir = output_dir
        self.seed = seed

        # routing
        distance = distance.lower().strip()
        assert distance in {"cosine", "euclidean"}
        self.distance = distance
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))

        # datamodule
        self.dm = datamodule  # may be None; we'll also check self.trainer.datamodule in setup()

        # prototype mode
        self.prototype_mode = prototype_mode.lower()
        assert self.prototype_mode in {"free", "example"}

        self.prototype_selector = (
            None if prototype_selector is None else prototype_selector.lower()
        )
        if self.prototype_mode == "free":
            self.prototype_selector = None

        # containers
        if self.prototype_mode == "free":
            self._prototype_bank_real = PrototypeBank(
                self.num_prototypes, hidden_dim
            )
            self.prototype_indices = None
            self.prototype_examples = None
            self.prototype_embeds = None
        else:
            self._prototype_bank_real = None
            self.prototype_indices: Optional[torch.LongTensor] = None
            # tokenized -> dict of tensors; raw-text -> {"text": list}
            self.prototype_examples: Optional[Dict[str, Any]] = None
            self.prototype_embeds: Optional[torch.Tensor] = None

        # metric buffers
        self._val_preds, self._val_labels = [], []
        self._test_preds, self._test_labels = [], []

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------- encoding & distances -------------
    def encode_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        if "text" in batch:
            embeds = self.encoder(batch["text"])
        else:
            kwargs = {"input_ids": batch["input_ids"]}
            if "attention_mask" in batch:
                kwargs["attention_mask"] = batch["attention_mask"]
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"]
            embeds = self.encoder(**kwargs)
        return embeds  # (B,H)

    def _current_prototype_matrix(self) -> torch.Tensor:
        """
        Unified accessor for the prototype matrix:
          - free mode -> learnable bank (P, H)
          - exemplar mode -> cached exemplar embeddings (P, H)
        """
        if self.prototype_mode == "free":
            return self._prototype_bank_real.weight
        else:
            assert (
                self.prototype_embeds is not None
            ), "Exemplar prototypes not set."
            return self.prototype_embeds

    def compute_distances(self, x_embed: torch.Tensor) -> torch.Tensor:
        """
        Always operate in normalized space.
        Cosine: d = 1 - <x, p>
        Euclidean: ||x - p||_2 over unit-normalized x,p (i.e., chord distance)
        """
        X = _normalize(x_embed)
        P = _normalize(self._current_prototype_matrix())
        if self.distance == "cosine":
            # cosine distance = 1 - cosine similarity
            # X: (B,H), P: (P,H) -> (B,P)
            return 1.0 - torch.einsum("bh,ph->bp", X, P)
        else:
            # Euclidean on unit sphere (equiv. monotone to cosine)
            return _cdist(X.unsqueeze(1), P.unsqueeze(0)).squeeze(1)

    # ------------- losses -------------
    def compute_losses(self, predictions, targets, x_embed):
        losses = {}
        total = torch.tensor(0.0, device=predictions.device)
        targets = targets.view_as(predictions).to(predictions.dtype)
        mse = F.mse_loss(predictions, targets)
        losses["mse"] = mse
        total = total + self.mse_weight * mse

        if self.cohesion_weight > 0.0:
            d = self.compute_distances(x_embed)
            min_d = d.min(dim=1).values
            cohesion = min_d.mean()
            losses["cohesion"] = cohesion
            total = total + self.cohesion_weight * cohesion

        if self.separation_weight > 0.0 and self.prototype_mode == "free":
            # separation on normalized prototypes
            P = _normalize(self._current_prototype_matrix())
            pairwise = torch.pdist(P, p=2)
            sep = (
                (-pairwise.mean())
                if pairwise.numel() > 0
                else torch.tensor(0.0, device=predictions.device)
            )
            losses["separation"] = sep
            total = total + self.separation_weight * sep

        losses["total"] = total
        return losses

    # ------------- head (override) -------------
    def predict_from_embeddings(self, embeds: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ------------- lightning steps -------------
    def _shared_step(self, batch, stage="val"):
        labels = batch["labels"].float()
        embeds = self.encode_batch(batch)
        preds = self.predict_from_embeddings(embeds).view(-1)
        losses = self.compute_losses(preds, labels, embeds)
        loss = losses["total"]

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
        else:
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

    # ------------- epoch metrics -------------
    def on_validation_epoch_start(self):
        self._val_preds, self._val_labels = [], []

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds)
        labels = torch.cat(self._val_labels)
        mse = tmf.mean_squared_error(preds, labels)
        rho = tmf.pearson_corrcoef(preds, labels)
        self.log("val_mse", mse, prog_bar=True)
        self.log("val_corr", rho, prog_bar=True)

    def on_test_epoch_start(self):
        self._test_preds, self._test_labels = [], []

    def on_test_epoch_end(self):
        if not self._test_preds:
            return
        preds = torch.cat(self._test_preds)
        labels = torch.cat(self._test_labels)
        mse = tmf.mean_squared_error(preds, labels)
        rho = tmf.pearson_corrcoef(preds, labels)
        self.log("test_mse", mse, prog_bar=True)
        self.log("test_corr", rho, prog_bar=True)

    # ---------- selection helpers ----------
    def _detect_mode_from_batch(self, batch: Dict[str, Any]) -> str:
        return "raw" if "text" in batch else "tokenized"

    def _resolve_dm(self):
        # prefer constructor-provided dm; fallback to trainer.datamodule if available
        return (
            self.dm
            if self.dm is not None
            else getattr(self.trainer, "datamodule", None)
        )

    def _make_ordered_train_loader(self, dm) -> DataLoader:
        """Create a NON-shuffled train dataloader so prototype indices are stable."""
        ds = dm.dataset["train"]
        collate = (
            None if getattr(dm, "tokenize_inputs", True) else dm._collate_raw
        )
        return DataLoader(
            ds,
            batch_size=dm.train_batch_size,
            shuffle=False,  # <-- critical: no shuffle
            collate_fn=collate,
        )

    @torch.no_grad()
    def _gather_train_embeddings_and_examples(
        self,
    ) -> Tuple[torch.Tensor, Dict[str, Any], str]:
        """
        Encode the entire train split ONCE (non-shuffled) to build (embeds_cpu, examples, mode),
        where examples are the minimal materials needed to re-encode selected exemplars later.
        """
        dm = self._resolve_dm()
        assert dm is not None, "datamodule required for exemplar selection."

        dl_ordered = self._make_ordered_train_loader(dm)
        first = next(iter(dl_ordered))
        mode = self._detect_mode_from_batch(first)
        # restart the ordered loader
        dl_ordered = self._make_ordered_train_loader(dm)

        device = self.device
        Bcat: Dict[str, List[torch.Tensor]] = {}
        texts: List[Union[str, Tuple[str, str]]] = []
        outs = []

        self.encoder.eval()
        print(
            "[proto:init] Collecting Training Representations for Prototype Initialization (no shuffle)"
        )
        for batch in tqdm(dl_ordered):
            if mode == "raw":
                emb = self.encode_batch({"text": batch["text"]}).detach()
                texts.extend([copy.deepcopy(t) for t in batch["text"]])
            else:
                enc = {"input_ids": batch["input_ids"].to(device)}
                if "attention_mask" in batch:
                    enc["attention_mask"] = batch["attention_mask"].to(device)
                if "token_type_ids" in batch:
                    enc["token_type_ids"] = batch["token_type_ids"].to(device)
                emb = self.encode_batch(enc).detach()
                for k in ("input_ids", "attention_mask", "token_type_ids"):
                    if k in batch:
                        Bcat.setdefault(k, [])
                        Bcat[k].append(batch[k].cpu().clone())
            outs.append(emb.cpu())

        if any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.train()

        embeds_cpu = torch.cat(outs, dim=0)  # raw on CPU
        examples = (
            {"text": texts}
            if mode == "raw"
            else {k: torch.cat(v, dim=0) for k, v in Bcat.items()}
        )
        return embeds_cpu, examples, mode

    @torch.no_grad()
    def _select_indices(
        self, embeds_cpu: torch.Tensor, k: int, method: Optional[str]
    ) -> torch.LongTensor:
        """
        Selection happens in the SAME normalized space used for training/routing.
        """
        N = embeds_cpu.size(0)
        if k >= N:
            return torch.arange(N, dtype=torch.long)

        embeds_cpu_n = _normalize(embeds_cpu)
        X = embeds_cpu_n.numpy()

        if method == "kmedoids":
            self.print(f"[proto:init] KMedoids select (k={k})", flush=True)
            km = KMedoids(
                n_clusters=k,
                metric="euclidean",
                method="alternate",
                init="k-medoids++",
                max_iter=300,
                random_state=self.seed,
            )
            km.fit(X)
            if (
                hasattr(km, "medoid_indices_")
                and km.medoid_indices_ is not None
            ):
                idx = torch.as_tensor(km.medoid_indices_, dtype=torch.long)
            elif (
                hasattr(km, "medoid_indices") and km.medoid_indices is not None
            ):
                idx = torch.as_tensor(km.medoid_indices, dtype=torch.long)
            else:
                centers = getattr(km, "cluster_centers_", None) or getattr(
                    km, "cluster_centers", None
                )
                if centers is not None:
                    idx = _nearest_point_indices(embeds_cpu_n, centers, k)
                else:
                    idx = _farthest_first(embeds_cpu_n, k)

            uniq = list(dict.fromkeys(idx.tolist()))
            if len(uniq) < k:
                extra = _farthest_first(
                    embeds_cpu_n, min(k, embeds_cpu_n.size(0))
                ).tolist()
                pool = [i for i in extra if i not in set(uniq)]
                uniq = uniq + pool[: (k - len(uniq))]
            return torch.tensor(uniq[:k], dtype=torch.long)

        if method == "kmeans++" or method is None:
            self.print(f"[proto:init] KMeans++ select (k={k})", flush=True)
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=k,  # <- as requested: n_init == k
                random_state=self.seed,
            )
            km.fit(X)
            centers = km.cluster_centers_
            return _nearest_point_indices(embeds_cpu_n, centers, k)

        if method == "farthest":
            self.print(
                f"[proto:init] Farthest-point select (k={k})", flush=True
            )
            return _farthest_first(embeds_cpu_n, k)

        if method == "dbscan":
            self.print("[proto:init] DBSCAN select", flush=True)
            db = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
            labels = db.fit_predict(X)
            if len([l for l in set(labels) if l != -1]) >= 1:
                idx = _cluster_means_to_indices(embeds_cpu_n, labels)
                if idx.numel() < k:
                    extra = _farthest_first(
                        embeds_cpu_n, min(k, embeds_cpu_n.size(0))
                    )
                    pool = [
                        i for i in extra.tolist() if i not in set(idx.tolist())
                    ]
                    idx = torch.tensor(
                        idx.tolist() + pool[: (k - idx.numel())],
                        dtype=torch.long,
                    )
                else:
                    idx = idx[:k]
                return idx
            return self._select_indices(embeds_cpu, k, "kmeans++")

        if method == "hdbscan":
            self.print("[proto:init] HDBSCAN select", flush=True)
            clusterer = HDBSCAN(min_cluster_size=10)
            labels = clusterer.fit_predict(X)
            if len([l for l in set(labels) if l != -1]) >= 1:
                idx = _cluster_means_to_indices(embeds_cpu_n, labels)
                if idx.numel() < k:
                    extra = _farthest_first(
                        embeds_cpu_n, min(k, embeds_cpu_n.size(0))
                    )
                    pool = [
                        i for i in extra.tolist() if i not in set(idx.tolist())
                    ]
                    idx = torch.tensor(
                        idx.tolist() + pool[: (k - idx.numel())],
                        dtype=torch.long,
                    )
                else:
                    idx = idx[:k]
                return idx
            return self._select_indices(embeds_cpu, k, "kmeans++")

        self.print(
            f"[proto:init] unknown selector '{method}', defaulting to KMeans++.",
            flush=True,
        )
        return self._select_indices(embeds_cpu, k, "kmeans++")

    @torch.no_grad()
    def _examples_to_chunk(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        """Build a model-ready batch chunk from cached exemplar examples."""
        if "text" in ex:
            return {"text": list(ex["text"])}  # new list (no alias)
        out = {"input_ids": ex["input_ids"].to(self.device)}
        if "attention_mask" in ex:
            out["attention_mask"] = ex["attention_mask"].to(self.device)
        if "token_type_ids" in ex:
            out["token_type_ids"] = ex["token_type_ids"].to(self.device)
        return out

    @torch.no_grad()
    def _set_example_prototypes(
        self, examples_full: Dict[str, Any], idx: torch.LongTensor
    ):
        """Store chosen examples + current embeddings (on model device)."""
        if "text" in examples_full:
            texts = [examples_full["text"][i] for i in idx.tolist()]
            ex = {"text": [copy.deepcopy(t) for t in texts]}
        else:
            ex = {k: v[idx].clone() for k, v in examples_full.items()}

        self.prototype_examples = ex
        emb = self.encode_batch(self._examples_to_chunk(ex)).detach()
        self.prototype_embeds = _normalize(emb).to(self.device)
        self.prototype_indices = idx.clone().cpu()

        self.print(
            f"[proto:init] set {idx.numel()} exemplar prototypes via '{self.prototype_selector}'. "
            f"first 10 idx: {idx[:10].tolist()}",
            flush=True,
        )

    @torch.no_grad()
    def _ensure_prototypes_ready(self):
        """If in exemplar mode and nothing is initialized yet, do a one-off selection now."""
        if self.prototype_mode != "example":
            return
        if (self.prototype_embeds is not None) or (
            self.prototype_indices is not None
        ):
            return
        print("[proto:init] Initializing Example-Constrained Prototypes")
        embeds_cpu, examples_full, _mode = (
            self._gather_train_embeddings_and_examples()
        )
        k = min(self.num_prototypes, embeds_cpu.size(0))
        idx = self._select_indices(embeds_cpu, k, self.prototype_selector)
        self._set_example_prototypes(examples_full, idx)

    def _maybe_move_prototypes_to_device(self):
        """Ensure prototype tensors live on the same device as the module."""
        dev = next(self.parameters()).device
        if (
            self.prototype_embeds is not None
            and self.prototype_embeds.device != dev
        ):
            self.prototype_embeds = self.prototype_embeds.to(dev)

    # ------------- trainer lifecycle hooks -------------
    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit", "validate", "test", "predict"):
            self._ensure_prototypes_ready()
            self._maybe_move_prototypes_to_device()

    def on_fit_start(self):
        self._ensure_prototypes_ready()
        self._maybe_move_prototypes_to_device()
        self.dump_trainable_status()
        self.print(
            f"[hparams] mode={self.prototype_mode}, selector={self.prototype_selector}, distance={self.distance}",
            flush=True,
        )

    def on_validation_start(self):
        self._ensure_prototypes_ready()
        self._maybe_move_prototypes_to_device()

    def on_test_start(self):
        self._ensure_prototypes_ready()
        self._maybe_move_prototypes_to_device()

    def on_after_backward(self):
        # If encoder is trainable & example mode: refresh ONLY the P prototype embeddings each step.
        if (
            self.prototype_mode == "example"
            and self.prototype_examples is not None
            and any(p.requires_grad for p in self.encoder.parameters())
        ):
            with torch.no_grad():
                chunk = self._examples_to_chunk(self.prototype_examples)
                emb = self.encode_batch(chunk)
                self.prototype_embeds = _normalize(emb.detach()).to(self.device)

    # ------------- persist/restore in checkpoints -------------
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["prototype_mode"] = self.prototype_mode
        checkpoint["num_prototypes"] = self.num_prototypes
        checkpoint["prototype_selector"] = self.prototype_selector
        checkpoint["distance"] = self.distance
        checkpoint["tau"] = self.tau.detach().cpu().item()

        if self.prototype_mode == "example":
            if self.prototype_embeds is not None:
                checkpoint["prototype_embeds"] = (
                    self.prototype_embeds.detach().cpu()
                )
            if self.prototype_indices is not None:
                checkpoint["prototype_indices"] = (
                    self.prototype_indices.detach().cpu()
                )

            # Persist examples so refresh keeps working if we resume training
            if self.prototype_examples is not None:
                ex = self.prototype_examples
                if "text" in ex:
                    checkpoint["prototype_examples_kind"] = "raw"
                    checkpoint["prototype_examples_text"] = list(ex["text"])
                else:
                    checkpoint["prototype_examples_kind"] = "tokenized"
                    for k in ("input_ids", "attention_mask", "token_type_ids"):
                        if k in ex:
                            checkpoint[f"prototype_examples_{k}"] = (
                                ex[k].detach().cpu().clone()
                            )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.prototype_mode = checkpoint.get(
            "prototype_mode", self.prototype_mode
        )
        self.num_prototypes = int(
            checkpoint.get("num_prototypes", self.num_prototypes)
        )
        self.prototype_selector = checkpoint.get(
            "prototype_selector", self.prototype_selector
        )
        self.distance = checkpoint.get("distance", self.distance)
        tau_val = checkpoint.get("tau", None)
        if tau_val is not None:
            with torch.no_grad():
                self.tau.fill_(float(tau_val))

        if self.prototype_mode == "example":
            pe = checkpoint.get("prototype_embeds", None)
            if pe is not None:
                self.prototype_embeds = pe
            pi = checkpoint.get("prototype_indices", None)
            if pi is not None:
                self.prototype_indices = pi

            kind = checkpoint.get("prototype_examples_kind", None)
            if kind == "raw":
                texts = checkpoint.get("prototype_examples_text", None)
                if texts is not None:
                    self.prototype_examples = {"text": list(texts)}
            elif kind == "tokenized":
                ex: Dict[str, torch.Tensor] = {}
                for k in ("input_ids", "attention_mask", "token_type_ids"):
                    t = checkpoint.get(f"prototype_examples_{k}", None)
                    if t is not None:
                        ex[k] = t
                if ex:
                    self.prototype_examples = ex

    # ------------- introspection -------------
    def dump_trainable_status(self):
        def cnt(m: nn.Module):
            tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
            tot = sum(p.numel() for p in m.parameters())
            return tr, tot

        lines = []
        if isinstance(self.encoder, nn.Module):
            tr, tot = cnt(self.encoder)
            lines.append(
                f"[trainable] encoder: {tr:,}/{tot:,} | training={self.encoder.training}"
            )
        else:
            lines.append("[trainable] encoder: (non-Module)")

        if self.prototype_mode == "free":
            trp = (
                self._prototype_bank_real.weight.numel()
                if self._prototype_bank_real.weight.requires_grad
                else 0
            )
            lines.append(
                f"[trainable] prototypes (free): {trp:,}/{self._prototype_bank_real.weight.numel():,}"
            )
        else:
            pcount = (
                0
                if self.prototype_embeds is None
                else self.prototype_embeds.numel()
            )
            lines.append(
                f"[trainable] prototypes (example): params=0 | cached embeds elems={pcount:,} "
                f"| indices set={self.prototype_indices is not None}"
            )

        total_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in self.parameters())
        lines.append(f"[trainable] TOTAL: {total_tr:,}/{total_all:,}")
        self.print("\n".join(lines), flush=True)

    # ------------- optim -------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
