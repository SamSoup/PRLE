# /model/encoder/SentenceEncoder.py  (with truncation)

import os
from typing import List, Tuple, Union
import torch
from torch import nn
from langchain_huggingface import HuggingFaceEmbeddings

TextPair = Union[
    Tuple[str, str], List[str]
]  # datasets may coerce tuples -> lists
TextInput = Union[List[str], List[TextPair]]


class SentenceEncoder(nn.Module):
    """
    Wrapper around HuggingFaceEmbeddings.

    Input (forward):
      - List[str]                      -> returns Tensor[B, D]
      - List[Tuple[str, str]] or List[List[str]] (len==2)
           encodes each element (u, v), then returns
             concat([u, v, u+v, |u-v|, u*v]) with shape Tensor[B, 5*D]
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str = "/scratch/06782/ysu707/.cache",
        normalize_embeddings: bool = False,
        batch_size: int = 32,
        max_tokens: int = 10000,  # 👈 cap very long sequences
    ):
        super().__init__()

        # Cache dirs
        os.environ["HF_HOME"] = cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_tokens = max_tokens

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": self.normalize_embeddings,
                "batch_size": self.batch_size,
            },
            multi_process=False,
            show_progress=False,
        )

        # Ensure pad token on the underlying tokenizer
        if self.embedding_model._client.tokenizer.pad_token is None:
            self.embedding_model._client.tokenizer.pad_token = (
                self.embedding_model._client.tokenizer.eos_token
            )

        # keep a handle to tokenizer
        self._tokenizer = self.embedding_model._client.tokenizer

    # ------------------------------------------------------------------
    # truncation helpers
    # ------------------------------------------------------------------
    def _truncate_text(self, text: str) -> str:
        """
        Tokenize -> cut to self.max_tokens -> decode back.
        This guarantees we never send >max_tokens into the ST model.
        """
        # normalize newlines early so token counts are stable
        text = text.replace("\n", " ")

        # encode without adding specials so we don't accidentally cut them off
        ids = self._tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(ids) <= self.max_tokens:
            return text

        # truncate
        ids = ids[: self.max_tokens]

        # decode back to string; skip_special_tokens so we don't reintroduce junk
        truncated = self._tokenizer.decode(ids, skip_special_tokens=True)
        return truncated

    def _truncate_texts(self, texts: List[str]) -> List[str]:
        return [self._truncate_text(t) for t in texts]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _embed_list(self, texts: List[str]) -> torch.Tensor:
        # 1) truncate long sequences
        texts = self._truncate_texts(texts)

        # 2) embed
        arr = self.embedding_model.embed_documents(texts)  # List[List[float]]

        # 3) turn into torch on the "main" device
        return torch.tensor(arr, device=self.device, dtype=torch.float32)

    def _pairwise_features(
        self, U: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([U, V, U + V, torch.abs(U - V), U * V], dim=1)

    def _looks_like_pair(self, x: object) -> bool:
        # Accept tuple or list of length 2 with string members
        if isinstance(x, (tuple, list)) and len(x) == 2:
            return isinstance(x[0], str) and isinstance(x[1], str)
        return False

    def forward(self, texts: TextInput) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty(0, 0, device=self.device)

        first = texts[0]
        if self._looks_like_pair(first):
            # Pair mode: truncate each side separately
            s1 = [self._truncate_text(t[0]) for t in texts]
            s2 = [self._truncate_text(t[1]) for t in texts]
            U = self._embed_list(s1)  # (B, D)
            V = self._embed_list(s2)  # (B, D)
            return self._pairwise_features(U, V)  # (B, 5D)
        else:
            # Single-string mode
            if not isinstance(first, str):
                texts = [str(t) for t in texts]
            return self._embed_list(texts)

    # Alias
    def encode(self, texts: TextInput) -> torch.Tensor:
        return self.forward(texts)
