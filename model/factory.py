# model/factory.py

from typing import Literal, Mapping, Any, Type

from .naive import NaiveExpertPRLE
from .linear import LinearExpertPRLE
from .mlp import MLPExpertPRLE  # <— new

# (future)
# from .sklearn import SklearnExpertPRLE

ModelType = Literal["naive", "linear", "mlp"]  # extend later with "sklearn"

# Registry pattern so adding new experts is just one line here.
_REGISTRY: Mapping[str, Type] = {
    "naive": NaiveExpertPRLE,
    "linear": LinearExpertPRLE,
    "mlp": MLPExpertPRLE,
    # "sklearn": SklearnExpertPRLE,
}


def get_model(
    model_type: ModelType,
    *,
    hidden_dim: int,
    num_prototypes: int,
    **kwargs: Any,
):
    """
    Factory for PRLE models.

    Args:
        model_type:
            Which expert architecture to use. Currently:
                - "naive": baseline expert variant
                - "linear": LinearExpertPRLE (1-layer linear regressor per prototype)
                - "mlp":   MLPExpertPRLE (tiny MLP per prototype)
              (Future: "sklearn", ...)

        hidden_dim:
            Dimensionality of retrieval-space embeddings (H). This becomes
            BasePrototypicalRegressor.hidden_dim.

        num_prototypes:
            How many prototypes / local experts to use (P).

        **kwargs:
            Forwarded directly into the model __init__. This includes everything
            BasePrototypicalRegressor (and subclasses) expect, for example:

            - lr
            - output_dir
            - datamodule
            - distance
            - seed
            - init_strategy
            - map_strategy
            - trainable_prototypes
            - use_value_space
            - proj_dim
            - em_alt_training
            - gating_strategy / knn_k / radius_threshold / mlp_hidden_dim
            - lambda_* loss weights
            - expert_init_strategy (for LinearExpertPRLE / MLPExpertPRLE)
            - mlp_hidden_dim (for MLPExpertPRLE)

    Returns:
        An instance of the chosen PRLE LightningModule subclass.
    """
    key = str(model_type).lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Available: {', '.join(_REGISTRY.keys())}"
        )

    cls = _REGISTRY[key]
    return cls(
        hidden_dim=hidden_dim,
        num_prototypes=num_prototypes,
        **kwargs,
    )
