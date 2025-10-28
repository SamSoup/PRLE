# train.py (updated for embedding-aware datamodule)

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
import wandb

from data.factory import get_datamodule
from data.embedding import EmbeddingDataModule
from model.factory import get_model
from model.encoder.factory import get_encoder


def _maybe_enable_tensor_cores(cfg=None):
    if not torch.cuda.is_available():
        return
    prec = None
    if cfg is not None:
        prec = (
            getattr(cfg.train, "float32_matmul_precision", None) or ""
        ).lower() or None
    if prec not in {"high", "medium", "highest"}:
        prec = "high"
    try:
        torch.set_float32_matmul_precision(prec)
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train(config):
    _maybe_enable_tensor_cores(config)

    # --------------------
    # Build the "raw" DataModule (text or tokenized)
    # --------------------
    encoder_type = str(
        config.model.encoder_type
    ).lower()  # e.g. "sentence" | "mean"
    tokenize_inputs = encoder_type != "sentence"

    dm_raw = get_datamodule(
        dataset_name=config.data.name,
        model_name=config.model.encoder_name,
        max_seq_length=config.data.max_seq_length,
        batch_size=config.train.train_batch_size,
        tokenize_inputs=tokenize_inputs,
        combine_fields=getattr(config.data, "combine_fields", False),
        combine_separator_token=getattr(
            config.data, "combine_separator_token", "[SEP]"
        ),
    )

    # --------------------
    # Build frozen encoder
    # --------------------
    encoder_kwargs = {}
    if encoder_type == "sentence":
        encoder_kwargs.update(
            cache_dir=getattr(config.model, "cache_dir", None),
            normalize_embeddings=getattr(
                config.model, "normalize_embeddings", False
            ),
            batch_size=config.train.train_batch_size,
        )
    if encoder_type == "mean":
        encoder_kwargs.update(
            cache_dir=getattr(config.model, "cache_dir", None),
            normalize=getattr(config.model, "normalize", False),
        )

    encoder = get_encoder(
        encoder_type=encoder_type,
        model_name=config.model.encoder_name,
        **encoder_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(encoder, torch.nn.Module):
        encoder.to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False  # freeze

    # --------------------
    # Wrap in EmbeddingDataModule
    # --------------------
    embed_dm = EmbeddingDataModule(
        raw_dm=dm_raw,
        encoder=encoder,
        embedding_cache_dir=getattr(
            config.data, "embedding_cache_dir", "embed_cache"
        ),
        train_batch_size=config.train.train_batch_size,
        eval_batch_size=getattr(config.train, "eval_batch_size", None),
        device=device,
    )

    # Let it prepare splits, compute/load embeddings, build its datasets
    embed_dm.setup("fit")
    embed_dm.setup("test")

    # --------------------
    # Build the model (LightningModule)
    # --------------------
    # Note: we pass datamodule=embed_dm so that BasePrototypicalRegressor
    # can access train_embeddings for prototype init.
    model = get_model(
        model_type=config.model.type,  # "linear", etc.
        encoder=encoder,
        hidden_dim=config.model.hidden_dim,
        num_prototypes=config.model.num_prototypes,
        mse_weight=getattr(config.model, "mse_weight", 1.0),
        cohesion_weight=getattr(config.model, "cohesion_weight", 0.0),
        separation_weight=getattr(config.model, "separation_weight", 0.0),
        lr=config.train.learning_rate,
        output_dir=getattr(config.model, "output_dir", "outputs"),
        freeze_encoder=True,  # encoder won't train
        datamodule=embed_dm,  # <-- embedding-based DM
        prototype_mode=getattr(config.model, "prototype_mode", "free"),
        prototype_selector=getattr(
            config.model, "prototype_selector", "kmeans++"
        ),
    )

    # --------------------
    # W&B loggers / callbacks
    # --------------------
    wandb_logger = WandbLogger(
        project=config.logging.project_name,
        config=OmegaConf.to_container(config, resolve=True),
        name=getattr(config.logging, "run_name", None),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-model",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=config.train.get("min_delta", 0.0),
        patience=config.train.get("early_stop_patience", 5),
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=config.train.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        deterministic=True,
    )

    # optional smoke test
    if getattr(config.train, "pretrain_smoke_test", False):
        print("⚡ Running test BEFORE training")
        trainer.test(model, datamodule=embed_dm, ckpt_path=None)

    # --------------------
    # Train
    # --------------------
    trainer.fit(model, datamodule=embed_dm, ckpt_path=None)

    # --------------------
    # Load Best Model
    # --------------------
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nLoading best model from: {best_model_path}\n")
    best_model = model.__class__.load_from_checkpoint(
        best_model_path,
        encoder=encoder,
    )

    # Reattach datamodule so best_model can access prototype info if needed
    best_model.dm = embed_dm

    # --------------------
    # Evaluate Best Model
    # --------------------
    trainer.test(best_model, datamodule=embed_dm)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PRLE models with cached-embedding datamodule."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/train.yaml",
        help="Path to an OmegaConf YAML config (default: configs/train.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    for seed in getattr(config.train, "seeds", [42]):
        pl.seed_everything(seed)

        if config.train.get("use_sweep", False):

            def sweep_train():
                with wandb.init():
                    config_sweep = OmegaConf.merge(config, wandb.config)
                    train(config_sweep)

            sweep_id = wandb.sweep(
                config.logging.sweep_config,
                project=config.logging.project_name,
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
