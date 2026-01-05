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
from model.encoders.factory import (
    get_encoder,
)  # used only to precompute embeddings

# NEW: stage-1 kernel
from model.kernel import KernelMetricLearner


def _maybe_enable_tensor_cores(cfg=None):
    """
    Enable TF32 math on Ampere+ GPUs for faster matmuls.
    """
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

    # -------------------------------------------------
    # 1. Build the "raw" DataModule (text/tokenized dataset)
    # -------------------------------------------------
    encoder_type = str(
        config.model.encoder_type
    ).lower()  # "sentence" | "mean" | ...
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

    # -------------------------------------------------
    # 2. Build / load the frozen encoder used for embedding extraction
    # -------------------------------------------------
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

    skip_encoder = getattr(config.model, "skip_loading_encoder", False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if skip_encoder:
        encoder = None
        print(
            "Skipping Encoder as requested. Please ensure embeddings are already computed in cache dir"
        )
    else:
        encoder = get_encoder(
            encoder_type=encoder_type,
            model_name=config.model.encoder_name,
            **encoder_kwargs,
        )

        if isinstance(encoder, torch.nn.Module):
            encoder.to(device)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False  # fully frozen

    # -------------------------------------------------
    # 3. Create the EmbeddingDataModule
    # -------------------------------------------------
    embed_dm = EmbeddingDataModule(
        raw_dm=dm_raw,
        encoder=encoder,
        embedding_cache_dir=getattr(
            config.data, "embedding_cache_dir", "embed_cache"
        ),
        train_batch_size=config.train.train_batch_size,
        eval_batch_size=getattr(
            config.train, "eval_batch_size", config.train.train_batch_size
        ),
        device=device,
    )

    embed_dm.setup("fit")
    embed_dm.setup("test")

    retrieval_dim = getattr(config.model, "hidden_dim", 1024)
    assert (
        retrieval_dim is not None
    ), "EmbeddingDataModule must define embedding_dim after setup()."

    # -------------------------------------------------
    # 3.5 Optional Stage 1: kernel metric learning (v2)
    # -------------------------------------------------
    pipeline_cfg = getattr(config, "pipeline", None)
    if pipeline_cfg is not None and hasattr(pipeline_cfg, "stages"):
        stages = list(pipeline_cfg.stages)
    else:
        stages = ["experts"]  # default: directly train experts

    kernel_ckpt_path = None
    if "kernel" in stages:
        kernel_cfg = getattr(config, "kernel", None)
        ckpt_dir = getattr(kernel_cfg, "checkpoint_dir", "checkpoints/kernel")
        os.makedirs(ckpt_dir, exist_ok=True)

        dataset_name = getattr(config.data, "name", "dataset")
        encoder_type_str = getattr(config.model, "encoder_type", "encoder")
        encoder_name_str = getattr(
            config.model, "encoder_name", "unknown"
        ).replace("/", "-")
        kernel_proj_type = getattr(
            kernel_cfg,
            "projection_type",
            getattr(config.model, "projection_type", "resmlp"),
        )

        kernel_ckpt_name = f"{dataset_name}_{encoder_type_str}_{encoder_name_str}_{kernel_proj_type}.ckpt"
        kernel_ckpt_path = os.path.join(ckpt_dir, kernel_ckpt_name)

        reuse_kernel = getattr(pipeline_cfg, "reuse_kernel", True)

        if not (reuse_kernel and os.path.exists(kernel_ckpt_path)):
            print(f"[kernel] training kernel from scratch → {kernel_ckpt_path}")

            # label_max comes from raw DataModule if available (e.g. STSB)
            label_max = getattr(dm_raw, "label_max", 1.0)

            kernel_proj_dim = getattr(
                kernel_cfg,
                "proj_dim",
                getattr(config.model, "proj_dim", None),
            )
            kernel_proj_kwargs = getattr(
                kernel_cfg,
                "projection_kwargs",
                getattr(config.model, "projection_kwargs", None),
            )

            kernel_lr = getattr(
                kernel_cfg, "learning_rate", config.train.learning_rate
            )
            kernel_max_epochs = getattr(kernel_cfg, "max_epochs", 5)

            lambda_pair = getattr(kernel_cfg, "lambda_pair", 1.0)
            lambda_triplet = getattr(kernel_cfg, "lambda_triplet", 0.0)
            lambda_uniform = getattr(kernel_cfg, "lambda_uniform", 0.01)

            kernel_model = KernelMetricLearner(
                hidden_dim=retrieval_dim,
                proj_dim=kernel_proj_dim,
                projection_type=kernel_proj_type,
                projection_kwargs=kernel_proj_kwargs,
                label_max=label_max,
                lambda_pair=lambda_pair,
                lambda_triplet=lambda_triplet,
                lambda_uniform=lambda_uniform,
                lr=kernel_lr,
            )

            kernel_trainer = pl.Trainer(
                max_epochs=kernel_max_epochs,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=(
                    torch.cuda.device_count()
                    if torch.cuda.is_available()
                    else 1
                ),
                logger=None,  # keep kernel training separate from main W&B run
                log_every_n_steps=10,
            )
            kernel_trainer.fit(kernel_model, datamodule=embed_dm)
            kernel_trainer.save_checkpoint(kernel_ckpt_path)
        else:
            print(
                f"[kernel] reusing existing kernel checkpoint: {kernel_ckpt_path}"
            )

    # -------------------------------------------------
    # 4. Build the PRLE model (LightningModule) - Stage 2
    # -------------------------------------------------
    model = get_model(
        model_type=config.model.type,  # e.g. "linear"
        hidden_dim=retrieval_dim,
        num_prototypes=config.model.num_prototypes,
        lr=config.train.learning_rate,
        output_dir=getattr(config.model, "output_dir", "outputs"),
        datamodule=embed_dm,
        # --- prototype / geometry config ---
        distance=getattr(config.model, "distance", "cosine"),
        seed=getattr(config.train, "seed", 42),
        init_strategy=getattr(config.model, "init_strategy", "random_real"),
        map_strategy=getattr(config.model, "map_strategy", "none"),
        trainable_prototypes=getattr(
            config.model, "trainable_prototypes", False
        ),
        # --- value space / metric learning ---
        use_value_space=getattr(config.model, "use_value_space", True),
        proj_dim=getattr(config.model, "proj_dim", None),
        em_alt_training=getattr(config.train, "em_alt_training", False),
        projection_type=getattr(config.model, "projection_type", "linear"),
        projection_kwargs=getattr(config.model, "projection_kwargs", None),
        # --- activation / routing config ---
        gating_strategy=getattr(config.model, "gating_strategy", "softmax"),
        knn_k=getattr(config.model, "knn_k", 3),
        radius_threshold=getattr(config.model, "radius_threshold", 0.5),
        activation_mlp_hidden_dim=getattr(
            config.model, "activation_mlp_hidden_dim", 64
        ),
        # --- loss lambda weights (geometry phase, v1) ---
        lambda_task_geom=getattr(config.loss, "lambda_task_geom", 1.0),
        lambda_metric_geom=getattr(config.loss, "lambda_metric_geom", 1.0),
        lambda_proto_self_geom=getattr(
            config.loss, "lambda_proto_self_geom", 0.0
        ),
        lambda_local_geom=getattr(config.loss, "lambda_local_geom", 0.0),
        lambda_consistency_geom=getattr(
            config.loss, "lambda_consistency_geom", 1.0
        ),
        # --- loss lambda weights (experts phase, v1) ---
        lambda_task_expert=getattr(config.loss, "lambda_task_expert", 1.0),
        lambda_metric_expert=getattr(config.loss, "lambda_metric_expert", 0.0),
        lambda_proto_self_expert=getattr(
            config.loss, "lambda_proto_self_expert", 1.0
        ),
        lambda_local_expert=getattr(config.loss, "lambda_local_expert", 1.0),
        lambda_consistency_expert=getattr(
            config.loss, "lambda_consistency_expert", 1.0
        ),
        # --- v2 loss config ---
        use_v2_loss=getattr(config.model, "use_v2_loss", False),
        lambda_task_main=getattr(config.loss, "lambda_task_main", 1.0),
        lambda_expert_fit=getattr(config.loss, "lambda_expert_fit", 0.5),
        lambda_anchor_route=getattr(config.loss, "lambda_anchor_route", 0.5),
        lambda_balance=getattr(config.loss, "lambda_balance", 0.1),
        lambda_entropy=getattr(config.loss, "lambda_entropy", 0.0),
        # --- (optional) expert init strategy for LinearExpertPRLE ---
        expert_init_strategy=getattr(
            config.model, "expert_init_strategy", "random"
        ),
        # --- (optional) hidden dim for MLPExpertPRLE ---
        expert_mlp_hidden_dim=config.model.get("expert_mlp_hidden_dim", 32),
    )

    # -------------------------------------------------
    # 4.5 Load kernel projection into PRLE model (if available)
    # -------------------------------------------------
    if (
        kernel_ckpt_path is not None
        and "kernel" in stages
        and getattr(config.model, "use_value_space", True)
        and os.path.exists(kernel_ckpt_path)
    ):
        print(
            f"[kernel→experts] loading projection head from {kernel_ckpt_path}"
        )
        ckpt_obj = torch.load(
            kernel_ckpt_path, map_location="cpu", weights_only=False
        )
        k_state = ckpt_obj["state_dict"]

        # KernelMetricLearner stores projection params under "projection_head.*"
        proj_state = {
            k[len("projection_head.") :]: v
            for k, v in k_state.items()
            if k.startswith("projection_head.")
        }

        pm = model.prototype_manager
        pm.projection_head.load_state_dict(proj_state, strict=False)
        # Freeze kernel during experts stage
        pm.enable_projection_training(False)

    # -------------------------------------------------
    # 5. W&B logging and Trainer setup
    # -------------------------------------------------
    wandb_logger = WandbLogger(
        project=config.logging.project_name,
        config=OmegaConf.to_container(config, resolve=True),
        name=getattr(config.logging, "run_name", None),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=getattr(config.model, "monitor", "val_mse"),
        mode=getattr(config.model, "mode", "min"),
        save_top_k=1,
        save_last=True,
        filename="best-model",
    )

    early_stop_callback = EarlyStopping(
        monitor=getattr(config.model, "monitor", "val_mse"),
        min_delta=config.train.get("min_delta", 0.0),
        patience=config.train.get("early_stop_patience", 5),
        verbose=True,
        mode=getattr(config.model, "mode", "min"),
    )

    trainer = pl.Trainer(
        max_epochs=config.train.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        deterministic=False,
    )

    # -------------------------------------------------
    # 6. (optional) smoke test BEFORE training
    # -------------------------------------------------
    if getattr(config.train, "pretrain_smoke_test", False):
        print("⚡ Running test BEFORE training")
        trainer.test(model, datamodule=embed_dm, ckpt_path=None)

    # -------------------------------------------------
    # 7. Fit
    # -------------------------------------------------
    trainer.fit(model, datamodule=embed_dm, ckpt_path=None)

    # -------------------------------------------------
    # 8. Eval best checkpoint
    # -------------------------------------------------
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nLoading best model from: {best_model_path}\n")

    best_model = model.__class__.load_from_checkpoint(
        best_model_path,
        strict=False,
        hidden_dim=retrieval_dim,
        num_prototypes=config.model.num_prototypes,
        lr=config.train.learning_rate,
        output_dir=getattr(config.model, "output_dir", "outputs"),
        datamodule=embed_dm,
        distance=getattr(config.model, "distance", "cosine"),
        seed=getattr(config.train, "seed", 42),
        init_strategy=getattr(config.model, "init_strategy", "random_real"),
        map_strategy=getattr(config.model, "map_strategy", "none"),
        trainable_prototypes=getattr(
            config.model, "trainable_prototypes", False
        ),
        use_value_space=getattr(config.model, "use_value_space", True),
        proj_dim=getattr(config.model, "proj_dim", None),
        em_alt_training=getattr(config.train, "em_alt_training", False),
        gating_strategy=getattr(config.model, "gating_strategy", "softmax"),
        knn_k=getattr(config.model, "knn_k", 3),
        radius_threshold=getattr(config.model, "radius_threshold", 0.5),
        activation_mlp_hidden_dim=getattr(
            config.model, "activation_mlp_hidden_dim", 64
        ),
        expert_mlp_hidden_dim=getattr(
            config.model, "expert_mlp_hidden_dim", 32
        ),
        lambda_task_geom=getattr(config.loss, "lambda_task_geom", 1.0),
        lambda_metric_geom=getattr(config.loss, "lambda_metric_geom", 1.0),
        lambda_proto_self_geom=getattr(
            config.loss, "lambda_proto_self_geom", 0.0
        ),
        lambda_local_geom=getattr(config.loss, "lambda_local_geom", 0.0),
        lambda_consistency_geom=getattr(
            config.loss, "lambda_consistency_geom", 1.0
        ),
        lambda_task_expert=getattr(config.loss, "lambda_task_expert", 1.0),
        lambda_metric_expert=getattr(config.loss, "lambda_metric_expert", 0.0),
        lambda_proto_self_expert=getattr(
            config.loss, "lambda_proto_self_expert", 1.0
        ),
        lambda_local_expert=getattr(config.loss, "lambda_local_expert", 1.0),
        lambda_consistency_expert=getattr(
            config.loss, "lambda_consistency_expert", 1.0
        ),
        expert_init_strategy=getattr(
            config.model, "expert_init_strategy", "random"
        ),
        projection_type=getattr(config.model, "projection_type", "linear"),
        projection_kwargs=getattr(config.model, "projection_kwargs", None),
        # v2 loss config for reload
        use_v2_loss=getattr(config.model, "use_v2_loss", False),
        lambda_task_main=getattr(config.loss, "lambda_task_main", 1.0),
        lambda_expert_fit=getattr(config.loss, "lambda_expert_fit", 0.5),
        lambda_anchor_route=getattr(config.loss, "lambda_anchor_route", 0.5),
        lambda_balance=getattr(config.loss, "lambda_balance", 0.1),
        lambda_entropy=getattr(config.loss, "lambda_entropy", 0.0),
    )

    ckpt_obj = torch.load(
        best_model_path, map_location="cpu", weights_only=False
    )
    state_dict = ckpt_obj["state_dict"]

    pm = best_model.prototype_manager

    buf_key = "prototype_manager._retrieval_prototypes_buf"
    if buf_key in state_dict:
        proto_buf = state_dict[buf_key].clone()
        if not hasattr(pm, "_retrieval_prototypes_buf"):
            pm.register_buffer(
                "_retrieval_prototypes_buf", proto_buf, persistent=True
            )
        else:
            getattr(pm, "_retrieval_prototypes_buf").data.copy_(proto_buf)
        pm._retrieval_prototypes_param = None

    idx_key = "prototype_manager.prototype_indices"
    if idx_key in state_dict:
        idx_buf = state_dict[idx_key].clone()
        if hasattr(pm, "prototype_indices"):
            pm.prototype_indices.data.copy_(idx_buf.long())
        else:
            pm.register_buffer(
                "prototype_indices", idx_buf.long(), persistent=True
            )

    pm._is_initialized = True
    if hasattr(pm, "_is_initialized_buf"):
        pm._is_initialized_buf.data.fill_(1)

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
