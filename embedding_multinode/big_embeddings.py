from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple, Union
import os

import torch
import torch.nn.functional as F
import deepspeed
from transformers import AutoTokenizer, AutoConfig, AutoModel
from accelerate import init_empty_weights
from tqdm import tqdm  # remove if you don't want a progress bar

TextLike = Union[str, Tuple[str, str]]
_VALID_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
}

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _to_torch_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{name}'. Choose one of {list(_DTYPE_MAP)}."
        )
    return _DTYPE_MAP[name]


def _dist_info() -> tuple[int, int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        rank, world = 0, 1
    local_rank = int(
        os.getenv("LOCAL_RANK", os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    )
    return rank, world, local_rank


class DeepSpeedShardedEmbeddings:
    """
    Multi-node tensor-parallel embeddings with DeepSpeed Inference.

    Uses META init (no weights) + DS 'checkpoint' JSON manifest to load sharded
    weights directly from disk: no full-model staging on any GPU.
    """

    def __init__(
        self,
        model_name: str,
        mp_size: Optional[int] = None,
        dtype: str = "bfloat16",
        max_length: Optional[int] = 4096,
        normalize: bool = False,
        pad_to_multiple_of: int = 8,
        trust_remote_code: bool = True,
        seed: Optional[int] = None,
        cache_dir: Optional[str] = None,  # tokenizer/config cache
        model_local_dir: Optional[
            str
        ] = None,  # local HF snapshot directory (REQUIRED)
        ds_manifest: Optional[
            str
        ] = None,  # path to DS JSON manifest (REQUIRED)
        progress: bool = False,
    ):
        import torch.distributed as dist

        rank, world_size, local_rank = _dist_info()
        self.rank = rank
        self.progress = progress

        # Device setup
        self.device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        self.dtype = _to_torch_dtype(dtype)
        self.max_length = max_length
        self.normalize = normalize
        self.pad_to_multiple_of = pad_to_multiple_of

        # Parallel size
        if mp_size is None:
            mp_size = world_size
        if mp_size != world_size:
            raise ValueError(
                f"mp_size ({mp_size}) must equal WORLD_SIZE ({world_size}) for tensor-parallel."
            )

        if seed is not None:
            torch.manual_seed(seed)

        # Required paths
        if not model_local_dir or not os.path.isdir(model_local_dir):
            raise FileNotFoundError(
                f"model_local_dir not found: {model_local_dir}"
            )
        if not ds_manifest or not os.path.isfile(ds_manifest):
            raise FileNotFoundError(
                f"DeepSpeed Inference JSON manifest not found: {ds_manifest}\n"
                f"Generate it with:\n"
                f"  bash embedding_multinode/make_ds_manifest.sh {model_local_dir} {ds_manifest} -t {mp_size} -d {dtype}"
            )

        # Tokenizer (normal load)
        tok_kwargs = dict(use_fast=True, trust_remote_code=trust_remote_code)
        if cache_dir:
            tok_kwargs["cache_dir"] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_local_dir, **tok_kwargs
        )

        # META init (no parameter allocation)
        cfg = AutoConfig.from_pretrained(
            model_local_dir,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        with init_empty_weights():
            base_model = AutoModel.from_config(
                cfg, trust_remote_code=trust_remote_code
            )

        # DeepSpeed Inference: load & shard weights USING THE MANIFEST
        ds_kwargs = dict(
            tensor_parallel={"tp_size": mp_size},
            dtype=self.dtype,
            replace_with_kernel_inject=True,
            checkpoint=ds_manifest,  # JSON file, not the directory
            base_dir=model_local_dir,  # base folder containing the shards
        )

        try:
            self.engine = deepspeed.init_inference(base_model, **ds_kwargs)
        except TypeError:
            # older DS signature
            ds_kwargs_fallback = dict(
                mp_size=mp_size,
                dtype=self.dtype,
                replace_with_kernel_inject=True,
                checkpoint=ds_manifest,
                base_dir=model_local_dir,
            )
            self.engine = deepspeed.init_inference(
                base_model, **ds_kwargs_fallback
            )

    # -------- Raw text path --------
    @torch.inference_mode()
    def _encode_batch(self, texts: List[TextLike]) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
        return self._forward_and_pool(enc)

    # -------- Tokenized path --------
    @torch.inference_mode()
    def encode_tokenized(
        self, token_batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        enc = {k: v for k, v in token_batch.items() if k in _VALID_INPUT_KEYS}
        missing = {"input_ids", "attention_mask"} - set(enc)
        if missing:
            raise ValueError(
                f"encode_tokenized() missing required keys: {missing}"
            )
        enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
        return self._forward_and_pool(enc)

    # -------- Internal --------
    @torch.inference_mode()
    def _forward_and_pool(self, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.engine(**enc, output_hidden_states=False, return_dict=True)
        last = (
            out.last_hidden_state
            if hasattr(out, "last_hidden_state")
            else out[0]
        )
        mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        embs = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        if self.normalize:
            embs = F.normalize(embs, p=2, dim=1)
        return embs

    # Convenience APIs
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode_iter(texts).cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._encode_batch([text]).cpu().tolist()[0]

    def _encode_iter(
        self, texts: Iterable[TextLike], batch_size: int = 32
    ) -> torch.Tensor:
        buf: List[TextLike] = []
        outs: List[torch.Tensor] = []
        iterator = (
            tqdm(texts, desc="Embedding")
            if (self.progress and self.rank == 0)
            else texts
        )
        for t in iterator:
            buf.append(t)
            if len(buf) >= batch_size:
                outs.append(self._encode_batch(buf))
                buf.clear()
        if buf:
            outs.append(self._encode_batch(buf))
        return torch.cat(outs, dim=0) if outs else torch.empty(0, 0)
