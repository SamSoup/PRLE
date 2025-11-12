# embedding_multinode/big_embeddings_zero3.py
# DeepSpeed 0.18.2-compatible ZeRO-3 inference loader for huge HF models (e.g., Qwen3-Next-80B)
# - Do NOT use zero.Init (0.18.2 lacks param_init_fn / init_device we need)
# - Load weights on CPU (materialized tensors, no meta), prevent GPU OOM
# - During deepspeed.initialize:
#     * Monkey-patch Module.to -> no-op (blocks massive CPU->CUDA move)
#     * Monkey-patch dist.broadcast -> no-op **for CPU tensors only**
#       (NCCL cannot broadcast CPU tensors; our ranks load identical weights)
# - World-size-aware DS config to satisfy batch-size assertion
# - Mean-pool + optional L2 normalize

import os
import json
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoConfig, AutoModel, AutoTokenizer


# --------------------------- helpers --------------------------- #


def _to_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _dtype_from_any(x: Any) -> torch.dtype:
    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        torch.bfloat16: torch.bfloat16,
        torch.float16: torch.float16,
        torch.float32: torch.float32,
    }
    return table.get(x, torch.bfloat16)


def _get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return _to_int(os.environ.get("WORLD_SIZE", 1), 1)


def _ensure_dist() -> int:
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = _to_int(os.environ.get("LOCAL_RANK", 0), 0)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return _get_world_size()


def _cleanup_dist() -> None:
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f) or {}


def _normalize_dtype_flags(cfg: Dict[str, Any], dtype: torch.dtype) -> None:
    # DS 0.18.2 recognizes fp16/bf16 blocks like this
    if dtype == torch.bfloat16:
        cfg["bf16"] = {"enabled": True}
        cfg.pop("fp16", None)
    elif dtype == torch.float16:
        cfg["fp16"] = {"enabled": True}
        cfg.pop("bf16", None)
    else:
        cfg["bf16"] = {"enabled": False}
        cfg["fp16"] = {"enabled": False}


def _worldsize_safe_ds_cfg(
    path: Optional[str],
    dtype: torch.dtype,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"DeepSpeed config not found: {path}")
        cfg = _load_json(path)

    zero = cfg.setdefault("zero_optimization", {})
    zero["stage"] = 3

    # Keep params off GPU; ZeRO-3 will page them as needed
    off = zero.setdefault("offload_param", {})
    off.setdefault("device", "cpu")
    off.setdefault("pin_memory", True)

    zero.setdefault("contiguous_gradients", False)
    zero.setdefault("overlap_comm", True)
    zero.setdefault("reduce_scatter", True)
    zero.setdefault("stage3_param_persistence_threshold", 0)
    zero.setdefault("stage3_max_live_parameters", 1e6)
    zero.setdefault("stage3_max_reuse_distance", 1e6)

    _normalize_dtype_flags(cfg, dtype)

    cfg["train_micro_batch_size_per_gpu"] = _to_int(
        cfg.get("train_micro_batch_size_per_gpu", 1), 1
    )
    cfg["gradient_accumulation_steps"] = _to_int(
        cfg.get("gradient_accumulation_steps", 1), 1
    )
    cfg.setdefault("gradient_clipping", 1.0)
    cfg.setdefault("prescale_gradients", False)
    return cfg


def _apply_batch_equation(cfg: Dict[str, Any], world_size: int) -> None:
    micro = _to_int(cfg.get("train_micro_batch_size_per_gpu", 1), 1)
    acc = _to_int(cfg.get("gradient_accumulation_steps", 1), 1)
    cfg["train_micro_batch_size_per_gpu"] = micro
    cfg["gradient_accumulation_steps"] = acc
    cfg["train_batch_size"] = micro * acc * max(1, world_size)


def _resolve_model_id(
    model_name: Optional[str], model: Optional[str], local_dir: Optional[str]
) -> str:
    if model_name and isinstance(model_name, str):
        return model_name
    if model and isinstance(model, str):
        return model
    if local_dir and isinstance(local_dir, str):
        return local_dir
    for k in ("MODEL_NAME", "HF_MODEL_ID", "MODEL"):
        v = os.environ.get(k)
        if v:
            return v
    raise ValueError(
        "Please provide `model_name` (or `model`) or a valid `model_local_dir`."
    )


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


# --------------------------- main wrapper ---------------------------- #


class DeepSpeedZero3Embeddings:
    """
    ZeRO-3 embedding runner for DeepSpeed 0.18.2:
    - Build HF model on CPU (materialized, no meta).
    - During DS engine init:
        * disable Module.to() to avoid CPU->CUDA mass move (OOM)
        * bypass dist.broadcast for CPU tensors (NCCL can’t handle CPU)
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        model: Optional[str] = None,
        dtype: Any = "bfloat16",
        max_length: Optional[int] = 4096,
        normalize: bool = False,
        cache_dir: Optional[str] = None,
        model_local_dir: Optional[str] = None,
        ds_config_path: Optional[
            str
        ] = "embedding_multinode/ds_zero3_inference.json",
        trust_remote_code: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        progress: bool = True,
        **kwargs: Any,
    ) -> None:
        # Distributed + env hygiene
        world_size = _ensure_dist()
        _dtype = _dtype_from_any(dtype)
        model_id = _resolve_model_id(model_name, model, model_local_dir)

        # DS config aware of real world size
        ds_cfg = _worldsize_safe_ds_cfg(ds_config_path, _dtype)
        _apply_batch_equation(ds_cfg, world_size)

        # Tokenizer
        tok_cache = model_local_dir or cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=tok_cache,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|pad|>"

        # HF config
        hf_config = AutoConfig.from_pretrained(
            model_id,
            cache_dir=tok_cache,
            trust_remote_code=trust_remote_code,
        )

        # --- Load the full model on CPU (real tensors, no meta) ---
        base_model = AutoModel.from_pretrained(
            model_id,
            config=hf_config,
            dtype=_dtype,
            cache_dir=tok_cache,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=False,  # important: materialize, don't go via meta
            device_map={"": "cpu"},  # force all weights to CPU
        )
        base_model.eval()

        # --- Guard against DS calling module.to(cuda) internally (OOM risk)
        # --- and guard against NCCL broadcast on CPU tensors
        orig_to = torch.nn.Module.to
        orig_broadcast = dist.broadcast

        def _noop_to(self, *args, **kwargs):
            # prevent massive CPU->CUDA move; DS will shard/offload parameters
            return self

        def _safe_broadcast(*args, **kwargs):
            """
            Wrap torch.distributed.broadcast to NO-OP for CPU tensors.
            Accept both positional and keyword usage to match c10d signature.
            """
            tensor = None
            if len(args) >= 1:
                tensor = args[0]
            if "tensor" in kwargs:
                tensor = kwargs["tensor"]
            if (
                tensor is not None
                and isinstance(tensor, torch.Tensor)
                and tensor.device.type == "cpu"
            ):
                # Ensure ranks rendezvous so we don't break ordering
                if dist.is_initialized():
                    dist.barrier()
                # async_op is False by default; returning None matches API
                return None
            return orig_broadcast(*args, **kwargs)

        try:
            torch.nn.Module.to = _noop_to
            dist.broadcast = _safe_broadcast
            engine, _, _, _ = deepspeed.initialize(
                model=base_model,
                model_parameters=[],  # inference-only
                config=ds_cfg,
                dist_init_required=False,  # already initialized
            )
        finally:
            torch.nn.Module.to = orig_to
            dist.broadcast = orig_broadcast

        self.engine = engine
        self.max_length = int(max_length) if max_length else 4096
        self.normalize = bool(normalize)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.dtype = _dtype

        # Reduce allocator fragmentation warnings if supported
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )

    # ------------------------ encode/pool path ------------------------ #

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        # Engine exposes its CUDA device
        device = getattr(self.engine, "device", torch.device("cuda"))
        return {k: v.to(device) for k, v in enc.items()}

    @torch.no_grad()
    def _forward_and_pool(self, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.engine(**enc, output_hidden_states=False, return_dict=True)
        hidden = out.last_hidden_state
        pooled = mean_pool(hidden, enc["attention_mask"])
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled

    @torch.no_grad()
    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        return self._forward_and_pool(self._tokenize(texts))

    @torch.no_grad()
    def _encode_iter(
        self, texts: Iterable[str], batch_size: int = 32
    ) -> torch.Tensor:
        buf: List[str] = []
        outs: List[torch.Tensor] = []
        for t in texts:
            buf.append(t)
            if len(buf) >= batch_size:
                outs.append(self._encode_batch(buf))
                buf.clear()
        if buf:
            outs.append(self._encode_batch(buf))
        return torch.cat(outs, dim=0) if outs else torch.empty(0, 0)

    @torch.no_grad()
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embs = self._encode_iter(texts).to(torch.float32).cpu().tolist()
        return embs

    def __del__(self):
        _cleanup_dist()
