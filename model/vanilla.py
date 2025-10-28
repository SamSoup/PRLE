# /model/vanilla.py
# A vanilla regression fine-tuner for HF models (tokenized inputs).
# - Optionally freeze the encoder (train head-only) or fine-tune end-to-end.
# - Logs train loss per step; logs val/test MSE & Pearson Corr per epoch.

from typing import List, Dict, Any
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics.functional as tmf
from transformers import AutoModelForSequenceClassification


class RegressionFinetuner(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float = 3e-5,
        weight_decay: float = 0.0,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )

        if freeze_encoder:
            self._freeze_encoder_params()
        else:
            # Unexplicitly Unfreeze everything by default
            for p in self.model.parameters():
                p.requires_grad = True

        # buffers for epoch metrics
        self.val_preds: List[torch.Tensor] = []
        self.val_labels: List[torch.Tensor] = []
        self.test_preds: List[torch.Tensor] = []
        self.test_labels: List[torch.Tensor] = []

    # ---- freezing utilities ----
    def _freeze_encoder_params(self):
        # Freeze everything by default
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze common head params across architectures
        for name, p in self.model.named_parameters():
            if any(
                k in name
                for k in [
                    "classifier",
                    "pre_classifier",
                    "score",
                    "regression",
                    "lm_head",
                ]
            ):
                p.requires_grad = True

        # Safety: if no head params found (rare), unfreeze final classifier module if exists
        if not any(p.requires_grad for p in self.model.parameters()):
            if hasattr(self.model, "classifier"):
                for p in self.model.classifier.parameters():
                    p.requires_grad = True

    # ---- forward & steps ----
    def forward(self, **batch: Dict[str, Any]) -> torch.Tensor:
        # Expect batch contains input_ids, attention_mask, (optional) token_type_ids
        out = self.model(
            **{k: v for k, v in batch.items() if k != "labels"},
            output_hidden_states=False,
        )
        logits = out.logits  # (B, 1)
        return logits.squeeze(-1)  # (B,)

    def training_step(self, batch, batch_idx):
        labels = batch["labels"].float()
        preds = self(**batch)
        loss = F.mse_loss(preds, labels)
        # log per step
        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True
        )
        # also aggregate at epoch end
        self.log(
            "train_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"].float()
        preds = self(**batch).detach()
        self.val_preds.append(preds)
        self.val_labels.append(labels)

    def on_validation_epoch_end(self):
        if self.val_preds:
            preds = torch.cat(self.val_preds).to(self.device)
            labels = torch.cat(self.val_labels).to(self.device)
            mse = tmf.mean_squared_error(preds, labels)
            rho = tmf.pearson_corrcoef(preds, labels)
            # epoch-only metrics; show in progress bar
            self.log(
                "val_mse",
                mse,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "val_corr",
                rho,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.val_preds.clear()
            self.val_labels.clear()

    def test_step(self, batch, batch_idx):
        labels = batch["labels"].float()
        preds = self(**batch).detach()
        self.test_preds.append(preds)
        self.test_labels.append(labels)

    def on_test_epoch_end(self):
        if self.test_preds:
            preds = torch.cat(self.test_preds).to(self.device)
            labels = torch.cat(self.test_labels).to(self.device)
            mse = tmf.mean_squared_error(preds, labels)
            rho = tmf.pearson_corrcoef(preds, labels)
            self.log(
                "test_mse",
                mse,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "test_corr",
                rho,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.test_preds.clear()
            self.test_labels.clear()

    # ---- optim ----
    def configure_optimizers(self):
        # Only optimize trainable params (respects freeze mode)
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer
