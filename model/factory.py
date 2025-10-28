from typing import Literal, Mapping, Any, Callable, Type
from .naive import NaiveExpertPRLE
from .linear import LinearExpertPRLE

# from .mlp import MLPExpertPRLE
# from .sklearn import SklearnExpertPRLE

ModelType = Literal["naive", "linear"]  # extend to "mlp", "sklearn" when added

# Registry pattern so adding new experts is just one line here.
_REGISTRY: Mapping[str, Type] = {
    "naive": NaiveExpertPRLE,
    "linear": LinearExpertPRLE,
    # "mlp": MLPExpertPRLE,
    # "sklearn": SklearnExpertPRLE,
}


def get_model(
    model_type: ModelType,
    *,
    encoder,
    hidden_dim: int,
    num_prototypes: int,
    **kwargs: Any,
):
    """
    Factory for PRLE models.

    Common kwargs (handled by BasePrototypicalRegressor):
      - mse_weight: float
      - cohesion_weight: float
      - separation_weight: float
      - lr: float

    Expert-specific kwargs (forwarded untouched):
      - (none for naive/linear)
      - for MLP (once enabled): mlp_layers: int, mlp_hidden: int
      - for sklearn (once enabled): expert_type: {"knn","svm","random_forest"}, etc.
    """
    key = str(model_type).lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Available: {', '.join(_REGISTRY.keys())}"
        )

    cls = _REGISTRY[key]
    return cls(
        encoder=encoder,
        hidden_dim=hidden_dim,
        num_prototypes=num_prototypes,
        **kwargs,
    )
