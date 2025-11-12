# embedding_multinode/big_embeddings_zero3.py
# Minimal, DeepSpeed 0.18.2-compatible ZeRO-3 embedder (no monkey patches)
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import os

import torch
import deepspeed
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.distributed as dist


def _as_dtype(x: Any) -> torch.dtype:
    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if isinstance(x, torch.dtype):
        return x
    return table.get(str(x).lower(), torch.bfloat16)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def _normalize_precision(ds_cfg: Dict[str, Any], dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16:
        ds_cfg["bf16"] = {"enabled": True}
        ds_cfg.pop("fp16", None)
    elif dtype == torch.float16:
        ds_cfg["fp16"] = {"enabled": True}
        ds_cfg.pop("bf16", None)
    else:
        ds_cfg["bf16"] = {"enabled": False}
        ds_cfg["fp16"] = {"enabled": False}


def _world_size() -> int:
    return (
        dist.get_world_size()
        if (dist.is_available() and dist.is_initialized())
        else 1
    )


def _apply_batch_equation(ds_cfg: Dict[str, Any], world: int) -> None:
    micro = int(ds_cfg.get("train_micro_batch_size_per_gpu", 1))
    acc = int(ds_cfg.get("gradient_accumulation_steps", 1))
    ds_cfg["train_micro_batch_size_per_gpu"] = micro
    ds_cfg["gradient_accumulation_steps"] = acc
    ds_cfg["train_batch_size"] = max(1, world) * micro * acc


class DeepSpeedZero3Embeddings:
    """
    Simple embedding runner:
    - Load HF model on CPU (materialized weights).
    - Initialize DeepSpeed ZeRO-3 with CPU offload (no tokenizer/model conversion step).
    - Mean-pool last_hidden_state; optional L2 normalize.
    """

    def __init__(
        self,
        *,
        model_id: str,
        model_local_dir: Optional[str] = None,
        ds_config_path: str,
        cache_dir: Optional[str] = None,
        dtype: Any = "bfloat16",
        max_length: int = 4096,
        normalize: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.dtype = _as_dtype(dtype)
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self.pad_to_multiple_of = pad_to_multiple_of

        model_ref = model_local_dir or model_id
        cache_ref = model_local_dir or cache_dir

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            cache_dir=cache_ref,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|pad|>"

        # Config
        hf_config = AutoConfig.from_pretrained(
            model_ref, cache_dir=cache_ref, trust_remote_code=trust_remote_code
        )

        # 1) Materialize model on CPU
        base_model = AutoModel.from_pretrained(
            model_ref,
            config=hf_config,
            dtype=self.dtype,
            cache_dir=cache_ref,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=False,  # real tensors, not meta
            device_map={"": "cpu"},
        )
        base_model.eval()

        # 2) Build ZeRO-3 config
        ds_cfg = _load_json(ds_config_path)
        zero = ds_cfg.setdefault("zero_optimization", {})
        zero["stage"] = 3
        off = zero.setdefault("offload_param", {})
        off.setdefault("device", "cpu")
        off.setdefault("pin_memory", True)
        zero.setdefault("overlap_comm", True)
        zero.setdefault("reduce_scatter", True)
        zero.setdefault("contiguous_gradients", False)
        zero.setdefault("stage3_param_persistence_threshold", 0)

        _normalize_precision(ds_cfg, self.dtype)
        _apply_batch_equation(ds_cfg, _world_size())

        # 3) Initialize DeepSpeed
        # NOTE: this will, try to load the whole model onto a node's local device
        # before trying to shard (even with cpu offload), which will explode
        # engine, _, _, _ = deepspeed.initialize(
        #     model=base_model,
        #     model_parameters=[],  # inference only
        #     config=ds_cfg,
        #     dist_init_required=False,  # caller already initialized process group
        # )

        # 3) Initialize DeepSpeed with two *scoped* guards: -- NOTE: this may take ~15m
        #    (a) No-op Module.to during init to avoid moving the full model to a single GPU
        #    (b) No-op dist.broadcast for CPU tensors (NCCL cannot broadcast CPU tensors)
        orig_to = torch.nn.Module.to
        import torch.distributed as dist

        orig_broadcast = dist.broadcast
        try:

            def _noop_to(self, *args, **kwargs):
                return self

            torch.nn.Module.to = _noop_to

            def _safe_broadcast(*args, **kwargs):
                tensor = None
                if len(args) >= 1:
                    tensor = args[0]
                if "tensor" in kwargs:
                    tensor = kwargs["tensor"]
                # If a CPU tensor is being broadcast on an NCCL pg, skip and keep ranks in lockstep.
                if (
                    isinstance(tensor, torch.Tensor)
                    and tensor.device.type == "cpu"
                ):
                    try:
                        if dist.is_initialized():
                            # device-aware barrier for NCCL default group; fall back otherwise
                            if (
                                dist.get_backend() == "nccl"
                                and torch.cuda.is_available()
                            ):
                                try:
                                    dist.barrier(
                                        device_ids=[torch.cuda.current_device()]
                                    )
                                except TypeError:
                                    dist.barrier()
                            else:
                                dist.barrier()
                    except Exception:
                        pass
                    return None
                return orig_broadcast(*args, **kwargs)

            dist.broadcast = _safe_broadcast

            engine, _, _, _ = deepspeed.initialize(
                model=base_model,
                model_parameters=[],  # inference only
                config=ds_cfg,
                dist_init_required=False,  # caller already initialized process group
            )
        finally:
            torch.nn.Module.to = orig_to
            dist.broadcast = orig_broadcast
        self.engine = engine

    # ---------- helpers ----------

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        device = getattr(self.engine, "device", torch.device("cuda"))
        return {k: v.to(device) for k, v in enc.items()}

    @torch.no_grad()
    def encode_tokenized(
        self, token_batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        out = self.engine(
            **token_batch, output_hidden_states=False, return_dict=True
        )
        pooled = self._mean_pool(
            out.last_hidden_state, token_batch["attention_mask"]
        ).to(torch.float32)
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled

    @torch.no_grad()
    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        return self.encode_tokenized(self._tokenize(texts))

    @torch.no_grad()
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        bs = 32
        outs = []
        for i in range(0, len(texts), bs):
            outs.append(self._encode_batch(texts[i : i + bs]))
        embs = torch.cat(outs, dim=0).cpu().to(torch.float32).tolist()
        return embs

    # --- diagnostics / cold-start precompile ---
    @torch.no_grad()
    def warmup(self, seq_len: int = 128) -> float:
        """
        Run one tiny forward pass to trigger kernel JIT, communicator setup,
        and ZeRO-3 sharding collects. Returns elapsed seconds.
        """
        device = getattr(self.engine, "device", torch.device("cuda"))
        dummy = ["warmup"] * 2
        enc = self.tokenizer(
            dummy,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        import time

        t0 = time.time()
        _ = self.encode_tokenized(enc)
        return float(time.time() - t0)
