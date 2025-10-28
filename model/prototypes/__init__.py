# model/prototypes/__init__.py

from .base import PrototypeManager
from .factory import build_prototype_manager

__all__ = [
    "PrototypeManager",
    "build_prototype_manager",
]
