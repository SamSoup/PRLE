# inference_multinode/datasets/__init__.py
from __future__ import annotations
from typing import Tuple, Any


def _load(name: str):
    key = name.lower()
    if key == "stsb":
        from . import stsb as mod

        return mod
    if key == "sickr_sts":
        from . import sickr_sts as mod

        return mod
    if key == "sts22_crosslingual":
        from . import sts22_crosslingual as mod

        return mod
    raise ValueError(f"Unknown dataset: {name}")


def make_module(
    name: str, *, limit: int | None, seed: int
) -> Tuple[Any, str, Any, Any]:
    """
    Returns: (dm, SYSTEM_MSG, build_user_message, parse_reply_to_0_1)
    where dm is an instance of dataset.InferenceModule(limit=..., seed=...)
    """
    mod = _load(name)
    dm = mod.InferenceModule(limit=limit, seed=seed)
    return dm, mod.SYSTEM_MSG, mod.build_user_message, mod.parse_reply_to_0_1
