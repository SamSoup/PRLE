# DeepSpeed 0.18.2 ZeRO-3 causal LM inference (multi-node, multi-GPU), local HF weights.
from __future__ import annotations
import os, json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _to_int(x: Any, d: int) -> int:
    try:
        return int(x)
    except Exception:
        return d


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
    return (
        dist.get_world_size()
        if (dist.is_available() and dist.is_initialized())
        else _to_int(os.environ.get("WORLD_SIZE", 1), 1)
    )


def _ensure_dist() -> int:
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )
        lr = _to_int(os.environ.get("LOCAL_RANK", 0), 0)
        if torch.cuda.is_available():
            torch.cuda.set_device(lr)
    return _get_world_size()


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f) or {}


def _normalize_dtype_flags(ds_cfg: Dict[str, Any], dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16:
        ds_cfg["bf16"] = {"enabled": True}
        ds_cfg.pop("fp16", None)
    elif dtype == torch.float16:
        ds_cfg["fp16"] = {"enabled": True}
        ds_cfg.pop("bf16", None)
    else:
        ds_cfg["bf16"] = {"enabled": False}
        ds_cfg["fp16"] = {"enabled": False}


def _worldsize_safe_ds_cfg(
    path: Optional[str], dtype: torch.dtype
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"DeepSpeed config not found: {path}")
        cfg = _load_json(path)
    zero = cfg.setdefault("zero_optimization", {})
    zero["stage"] = 3
    zero.setdefault("overlap_comm", True)
    zero.setdefault("reduce_scatter", True)
    zero.setdefault("contiguous_gradients", False)
    off = zero.setdefault("offload_param", {})
    off.setdefault("device", "none")  # shard in VRAM by default
    _normalize_dtype_flags(cfg, dtype)
    cfg.setdefault("train_micro_batch_size_per_gpu", 1)
    cfg.setdefault("gradient_accumulation_steps", 1)
    cfg.setdefault("prescale_gradients", False)
    return cfg


class DeepSpeedZero3LM:
    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        model_local_dir: Optional[str] = None,
        dtype: Any = "bfloat16",
        max_length: int = 512,
        ds_config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        _ensure_dist()
        self.dtype = _dtype_from_any(dtype)
        model_ref = model_local_dir or model_id
        if not model_ref:
            raise ValueError("Provide model_id or model_local_dir")

        tok_cache = model_local_dir or cache_dir

        hf_config = AutoConfig.from_pretrained(
            model_ref, cache_dir=tok_cache, trust_remote_code=trust_remote_code
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            cache_dir=tok_cache,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|pad|>"

        # left padding for decoder-only architectures
        if not getattr(hf_config, "is_encoder_decoder", False):
            # decoder-only → left pad
            self.tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            config=hf_config,
            dtype=self.dtype,
            cache_dir=tok_cache,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=False,
            device_map={"": "cpu"},  # avoid accidental full cuda()
        )
        base_model.eval()

        orig_to = torch.nn.Module.to
        orig_broadcast = dist.broadcast

        def _noop_to(self_mod, *args, **kwargs):
            return self_mod

        def _safe_broadcast(*args, **kwargs):
            tensor = kwargs.get("tensor", args[0] if args else None)
            if (
                tensor is not None
                and isinstance(tensor, torch.Tensor)
                and tensor.device.type == "cpu"
            ):
                if dist.is_initialized():
                    dist.barrier()
                return None
            return orig_broadcast(*args, **kwargs)

        try:
            torch.nn.Module.to = _noop_to
            dist.broadcast = _safe_broadcast
            ds_cfg = _worldsize_safe_ds_cfg(ds_config_path, self.dtype)
            engine, _, _, _ = deepspeed.initialize(
                model=base_model,
                model_parameters=[],
                config=ds_cfg,
                dist_init_required=False,
            )
        finally:
            torch.nn.Module.to = orig_to
            dist.broadcast = orig_broadcast

        self.engine = engine
        self.max_length = int(max_length)
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )

    @torch.no_grad()
    def warmup_generate(self, max_new_tokens: int = 8) -> None:
        prompt = "Hello"
        toks = self.tokenizer([prompt], return_tensors="pt", padding=True).to(
            self.engine.device
        )
        _ = self.engine.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic warmup
            use_cache=True,
        )

    @torch.no_grad()
    def batch_chat(
        self,
        *,
        system_messages: List[str],
        user_messages: List[str],
        max_new_tokens: int,
        temperature: float,
    ) -> List[str]:
        """
        Batched chat-style generation. Returns list of decoded completions (no prompt).
        """
        assert len(system_messages) == len(user_messages)
        B = len(user_messages)
        texts: List[str] = []
        for i in range(B):
            msgs = [
                {"role": "system", "content": system_messages[i]},
                {"role": "user", "content": user_messages[i]},
            ]
            try:
                t = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                t = (system_messages[i] + "\n\n" + user_messages[i]).strip()
            texts.append(t)

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(self.engine.device) for k, v in enc.items()}

        # Only pass temperature when sampling; some models ignore/complain otherwise.
        gen_kwargs = dict(max_new_tokens=max_new_tokens, use_cache=True)
        if temperature and temperature > 0.0:
            gen_kwargs.update(
                dict(do_sample=True, temperature=float(max(1e-5, temperature)))
            )
        else:
            gen_kwargs.update(dict(do_sample=False))

        out = self.engine.generate(**enc, **gen_kwargs)  # [B, T_out]
        input_lens = enc["attention_mask"].sum(dim=1).tolist()

        replies: List[str] = []
        for i in range(B):
            gen_ids = out[i, input_lens[i] :]
            raw = self.tokenizer.decode(
                gen_ids, skip_special_tokens=True
            ).strip()
            replies.append(raw)
        return replies
