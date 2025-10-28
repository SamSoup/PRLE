from typing import Optional
import pytorch_lightning as pl
from .base import PrototypeManager, InitStrategy, MapStrategy, DistanceMetric


def build_prototype_manager(
    *,
    num_prototypes: int,
    retrieval_dim: int,
    proj_dim: Optional[int],
    init_strategy: InitStrategy,
    map_strategy: MapStrategy,
    distance_metric: DistanceMetric,
    trainable_prototypes: bool,
    use_value_space: bool,
    seed: int = 42,
) -> PrototypeManager:
    """
    Factory for PrototypeManager, wires in use_value_space so the caller
    doesn't have to think about identity vs learned projection head.
    """
    manager = PrototypeManager(
        num_prototypes=num_prototypes,
        retrieval_dim=retrieval_dim,
        proj_dim=proj_dim,
        init_strategy=init_strategy,
        map_strategy=map_strategy,
        distance_metric=distance_metric,
        trainable_prototypes=trainable_prototypes,
        use_value_space=use_value_space,
        seed=seed,
    )
    return manager
