# train.py
# Minimal edits of your existing train.py to run the *vanilla* finetuner
# (AutoModelForSequenceClassification for regression) with W&B logging.
#
# Differences vs PRLE version:
# - No custom encoder/model factory: we use RegressionFinetuner (model/vanilla.py)
# - We force tokenize_inputs=True in the DataModule
# - We monitor "val_mse" (lower is better) for checkpointing/early stopping

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
import wandb

from data.factory import get_datamodule
from model.vanilla import RegressionFinetuner


def train(config):
    # --------------------
    # Load Dataset (tokenized for vanilla HF finetune)
    # --------------------
    dm = get_datamodule(
        dataset_name=config.data.name,
        model_name=config.model.name,  # HF model name
        max_seq_length=config.data.max_seq_length,
        batch_size=config.train.train_batch_size,
        tokenize_inputs=True,  # <— vanilla finetune expects tokenized inputs
        combine_fields=getattr(config.data, "combine_fields", False),
        combine_separator_token=getattr(
            config.data, "combine_separator_token", "[SEP]"
        ),
    )
    dm.setup("fit")

    # --------------------
    # Build Model (LightningModule)
    # --------------------
    model = RegressionFinetuner(
        model_name=config.model.name,
        lr=config.train.learning_rate,
        weight_decay=getattr(config.train, "weight_decay", 0.0),
        freeze_encoder=bool(getattr(config.model, "freeze_encoder", False)),
    )

    # --------------------
    # W&B Logging & Callbacks
    # --------------------
    wandb_logger = WandbLogger(
        project=config.logging.project_name,
        config=OmegaConf.to_container(config, resolve=True),
        name=getattr(config.logging, "run_name", None),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_mse",  # <— monitor MSE (minimize)
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-model",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_mse",  # <— stop on MSE
        min_delta=config.train.get("min_delta", 0.0),
        patience=config.train.get("early_stop_patience", 5),
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=config.train.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=getattr(config.train, "log_every_n_steps", 10),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        deterministic=True,
    )

    # --------------------
    # (optional) quick smoke test BEFORE training
    # --------------------
    if getattr(config.train, "pretrain_smoke_test", False):
        print("⚡ Running test BEFORE fine-tuning")
        trainer.test(model, datamodule=dm, ckpt_path=None)

    # --------------------
    # Train
    # --------------------
    trainer.fit(model, datamodule=dm, ckpt_path=None)

    # --------------------
    # Load Best Model
    # --------------------
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nLoading best model from: {best_model_path}\n")
    best_model = RegressionFinetuner.load_from_checkpoint(
        best_model_path,
        model_name=config.model.name,
        lr=config.train.learning_rate,
        weight_decay=getattr(config.train, "weight_decay", 0.0),
        freeze_encoder=bool(getattr(config.model, "freeze_encoder", False)),
    )

    # --------------------
    # Evaluate Best Model
    # --------------------
    trainer.test(best_model, datamodule=dm)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vanilla regression fine-tuning with W&B logging."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/vanilla_stsb.yaml",
        help="Path to an OmegaConf YAML config (default: configs/vanilla_stsb.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    for seed in getattr(config.train, "seeds", [42]):
        # --------------------
        # Reproducibility
        # --------------------
        pl.seed_everything(seed)

        # ---- W&B Sweep (optional) ----
        if config.train.get("use_sweep", False):
            # 1) Convert DictConfig -> dict for wandb.sweep
            sweep_config = OmegaConf.to_container(
                config.logging.sweep_config, resolve=True
            )

            def sweep_train():
                with wandb.init():
                    # 2) Merge wandb.config (a simple dict-like) back into an OmegaConf
                    wb_config = OmegaConf.create(dict(wandb.config))
                    config_sweep = OmegaConf.merge(config, wb_config)
                    train(config_sweep)

            sweep_id = wandb.sweep(
                sweep=sweep_config, project=config.logging.project_name
            )
            wandb.agent(
                sweep_id,
                function=sweep_train,
                count=config.logging.sweep_trials,
            )
        else:
            train(config)


if __name__ == "__main__":
    main()
