# model/prototypes/base.py

from typing import Optional, Tuple, Literal
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans


InitStrategy = Literal[
    "random",  # random Gaussian vectors, not tied to real examples
    "random_real",  # sample real training points
    "k-medoids",  # medoid representatives (real training points)
    "k-means++",  # k-means++ centers snapped to nearest real point
    "farthest-first",  # greedy k-center cover on cosine distance
]

MapStrategy = Literal[
    "none",
    "projection",  # DURING TRAINING: periodically snap drifting trainable
    # prototypes back to nearest real examples
]

DistanceMetric = Literal["cosine", "euclidean"]


def _normalize(x: Tensor) -> Tensor:
    """Row-wise L2 normalize."""
    return F.normalize(x, p=2, dim=-1)


def _normalize_strategy(name: str) -> str:
    """
    Accept multiple spellings from configs and map them to canonical names.
    e.g. "kmeans++" -> "k-means++", "farthest_first" -> "farthest-first"
    """
    s = name.strip().lower().replace(" ", "")
    aliases = {
        "kmeans++": "k-means++",
        "k-means++": "k-means++",
        "k_medoids": "k-medoids",
        "kmedoids": "k-medoids",
        "k-medoids": "k-medoids",
        "farthest_first": "farthest-first",
        "farthestfirst": "farthest-first",
        "farthest-first": "farthest-first",
        "random_real": "random_real",
        "random-real": "random_real",
        "random": "random",
    }
    return aliases.get(s, name)


class PrototypeManager(nn.Module):
    """
    Manages:
      - Retrieval-space prototypes of shape (P, H_retrieval).
        These are either:
            * a trainable nn.Parameter (if trainable_prototypes=True), OR
            * a frozen registered buffer (if trainable_prototypes=False)

      - A projection head g(·) mapping retrieval space -> value space.
        If use_value_space=True: learned Linear(H -> proj_dim)
        If use_value_space=False: nn.Identity()

      - prototype_indices: (P,) buffer of dataset row IDs for interpretability.

      - periodic_snap(): optional snapping back to nearest real point during
        training if prototypes are trainable and map_strategy == "projection".

      - enable_projection_training() / enable_prototype_training():
        knobs for EM-style alternating optimization.

    We also track `_is_initialized_buf` (uint8 {0,1}) so we know if the real
    prototypes have been populated yet. This is crucial because we now
    *always* register a placeholder buffer in __init__ for checkpoint safety.
    """

    def __init__(
        self,
        num_prototypes: int,
        retrieval_dim: int,
        proj_dim: Optional[int],
        init_strategy: InitStrategy,
        map_strategy: MapStrategy,
        distance_metric: DistanceMetric,
        trainable_prototypes: bool,
        use_value_space: bool,
        seed: int = 42,
    ):
        super().__init__()

        # --- core config ---
        self.num_prototypes = int(num_prototypes)
        self.retrieval_dim = retrieval_dim
        self.use_value_space = bool(use_value_space)

        # if we are NOT using a separate value space, proj_dim defaults to retrieval_dim
        self.proj_dim = proj_dim if (proj_dim is not None) else retrieval_dim

        # normalize init strategy aliases (kmeans++ vs k-means++, etc.)
        self.init_strategy = _normalize_strategy(init_strategy)
        self.map_strategy = map_strategy
        self.distance_metric = distance_metric
        self.trainable_prototypes = bool(trainable_prototypes)
        self.seed = seed

        # prototype_indices: (P,)
        # ties each prototype to a training row index (or -1 if synthetic)
        # Register as a buffer so it's in checkpoints.
        self.register_buffer(
            "prototype_indices",
            torch.full(
                (self.num_prototypes,),
                fill_value=-1,
                dtype=torch.long,
            ),
            persistent=True,
        )

        # Track initialized flag as a buffer so it survives checkpoints.
        # 0 -> not yet initialized from real train embeddings
        # 1 -> initialized
        self.register_buffer(
            "_is_initialized_buf",
            torch.tensor(0, dtype=torch.uint8),
            persistent=True,
        )
        self._is_initialized = False  # local convenience mirror

        # Placeholders for prototype storage:
        #
        # We ALWAYS register a buffer `_retrieval_prototypes_buf` of the final
        # target shape (P, H). This guarantees:
        #   - state_dict() always contains this key
        #   - load_from_checkpoint() will find a matching key/shape
        #
        # If trainable_prototypes is False (buffer mode):
        #   - we'll later OVERWRITE this buffer with the real prototype vectors.
        #
        # If trainable_prototypes is True (param mode):
        #   - we'll create `_retrieval_prototypes_param` as an nn.Parameter
        #     in initialize_from_train_embeddings(), and we IGNORE the buffer
        #     at inference time.
        #
        # The buffer starts as zeros but that's fine before init.
        self.register_buffer(
            "_retrieval_prototypes_buf",
            torch.zeros(
                self.num_prototypes,
                self.retrieval_dim,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self._retrieval_prototypes_param: Optional[nn.Parameter] = None

        # --- projection head g(·) ---
        if self.use_value_space:
            # trainable linear projection
            self.projection_head: nn.Module = nn.Linear(
                self.retrieval_dim,
                self.proj_dim,
                bias=False,
            )
            self._projection_trainable = True
        else:
            # identity mapping, no params
            self.projection_head = nn.Identity()
            self._projection_trainable = False

        self._set_seed()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    @torch.no_grad()
    def initialize_from_train_embeddings(
        self,
        train_embeds: Tensor,  # (N, H_retrieval) on CPU
    ) -> None:
        """
        One-time setup before training:
          1) pick P prototype anchors using init_strategy
          2) store them as either nn.Parameter or buffer
          3) record which training rows they came from
          4) mark _is_initialized_buf = 1
        """
        if self._is_initialized or int(self._is_initialized_buf.item()) == 1:
            # already initialized (fresh run or loaded from checkpoint)
            return

        proto_idx, proto_vecs = self._init_raw_prototypes(train_embeds)
        # proto_idx:  (P,)
        # proto_vecs: (P,H)

        self._register_retrieval_prototypes(proto_vecs)

        # record which training example each prototype is tied to
        self.prototype_indices.data.copy_(proto_idx.long())

        self._is_initialized = True
        self._is_initialized_buf.data.copy_(torch.tensor(1, dtype=torch.uint8))

    def periodic_snap(
        self,
        train_embeds: Tensor,  # (N,H), typically CPU
    ) -> None:
        """
        Training-time interpretability tether.

        If trainable_prototypes==True AND map_strategy=="projection",
        then after each epoch:
          - find nearest real training embedding for each prototype
          - overwrite prototype vectors with those real embeddings
          - update prototype_indices accordingly

        If prototypes are frozen or map_strategy=="none", no-op.
        """
        if not self.trainable_prototypes:
            return
        if self.map_strategy != "projection":
            return
        if self._retrieval_prototypes_param is None:
            return

        self._maybe_snap_to_real_points(train_embeds)

    # ------------------------------------------------------------------
    # Accessors (device-aware)
    # ------------------------------------------------------------------

    def _fallback_device(self) -> torch.device:
        """
        Pick a 'best guess' device to return prototypes on.
        Priority:
          1. any trainable parameter
          2. any registered buffer
          3. cpu
        """
        # try parameters first (includes projection_head params etc.)
        for p in self.parameters(recurse=True):
            return p.device
        # then buffers
        for _, b in self.named_buffers(recurse=True):
            return b.device
        return torch.device("cpu")

    def get_retrieval_prototypes(
        self,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Return retrieval-space prototypes (P,H) on the requested device.
        If device is None, infer a sensible device.
        """
        if device is None:
            device = self._fallback_device()

        if self.trainable_prototypes:
            assert (
                self._retrieval_prototypes_param is not None
            ), "trainable prototypes requested but param is missing"
            return self._retrieval_prototypes_param.to(device)
        else:
            return self._retrieval_prototypes_buf.to(device)

    def project_embeddings(self, x: Tensor) -> Tensor:
        """
        Map retrieval embeddings -> VALUE space.

        If use_value_space=True, this is a learned Linear.
        If use_value_space=False, projection_head is Identity() and just
        returns x unchanged.
        """
        return self.projection_head(x)

    def get_value_prototypes(
        self,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        VALUE-space prototypes g(proto).
        We compute them on the requested device for consistency with callers.
        """
        with torch.no_grad():
            rp = self.get_retrieval_prototypes(device=device)  # (P,H)
            vp = self.project_embeddings(rp)  # (P,Dv)
        return vp

    # ------------------------------------------------------------------
    # EM-style knobs
    # ------------------------------------------------------------------

    def enable_projection_training(self, flag: bool) -> None:
        """
        Toggle gradient updates for the projection head g(·).
        Only meaningful if use_value_space=True.
        """
        if not self.use_value_space:
            # identity head, nothing to train
            self._projection_trainable = False
            return

        self._projection_trainable = bool(flag)
        self.projection_head.train(flag)
        for p in self.projection_head.parameters():
            p.requires_grad = flag

    def enable_prototype_training(self, flag: bool) -> None:
        """
        Toggle gradient updates for the prototype vectors themselves.
        Only relevant if trainable_prototypes=True (nn.Parameter mode).
        """
        if not self.trainable_prototypes:
            return
        if self._retrieval_prototypes_param is None:
            return
        self._retrieval_prototypes_param.requires_grad_(flag)

    # ------------------------------------------------------------------
    # Metric learning loss (VALUE space)
    # ------------------------------------------------------------------

    def metric_learning_loss(
        self,
        value_embeds: Tensor,  # (B, proj_dim)
        labels: Tensor,  # (B,)
    ) -> Tensor:
        """
        Encourage similar labels to be close in value space, and dissimilar
        labels to be apart.

        If we are not training the projection head right now (i.e. we're
        in "experts" phase or use_value_space=False), return 0.
        """
        # If we aren't even *using* a distinct value space, skip.
        if not self.use_value_space:
            return torch.zeros((), device=value_embeds.device)

        # If projection head is currently frozen, skip.
        if not self._projection_trainable:
            return torch.zeros((), device=value_embeds.device)

        if value_embeds.size(0) < 2:
            return torch.zeros((), device=value_embeds.device)

        # hyperparams for contrastive-style shaping
        pos_thresh = 0.1
        neg_thresh = 0.3
        margin = 1.0

        z = F.normalize(value_embeds, p=2, dim=-1)  # (B,D)
        B = z.size(0)

        dists = torch.cdist(z, z, p=2)  # (B,B)

        y = labels.view(B, 1)
        ydiff = (y - y.T).abs()  # (B,B)

        eye = torch.eye(B, dtype=torch.bool, device=z.device)

        pos_mask = (ydiff <= pos_thresh) & (~eye)
        neg_mask = (ydiff >= neg_thresh) & (~eye)

        if pos_mask.any():
            pos_d = dists[pos_mask]
            pos_loss = (pos_d**2).mean()
        else:
            pos_loss = torch.zeros((), device=z.device)

        if neg_mask.any():
            neg_d = dists[neg_mask]
            neg_loss = (F.relu(margin - neg_d) ** 2).mean()
        else:
            neg_loss = torch.zeros((), device=z.device)

        return pos_loss + neg_loss

    # ------------------------------------------------------------------
    # Future expansion (not implemented)
    # ------------------------------------------------------------------

    def replace_or_prune_prototypes(
        self,
        train_embeds: Tensor,
        train_labels: Optional[Tensor] = None,
        usage_stats: Optional[Tensor] = None,
    ) -> None:
        raise NotImplementedError(
            "prototype pruning / replacement is not implemented yet."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _register_retrieval_prototypes(self, proto_vectors: Tensor) -> None:
        """
        After init: store prototype vectors (P,H_retrieval) either as
        a trainable nn.Parameter or copy them into the pre-registered buffer.

        We DO NOT call register_buffer here (already done in __init__)
        to avoid KeyError and to keep checkpoint loading consistent.
        """
        proto_vectors = proto_vectors.float()

        if self.trainable_prototypes:
            # trainable path
            self._retrieval_prototypes_param = nn.Parameter(proto_vectors)
            # we leave _retrieval_prototypes_buf alone (it's unused at runtime
            # in this mode but stays in the state_dict for shape consistency)
        else:
            # frozen/buffer path: just overwrite the placeholder buffer in-place
            with torch.no_grad():
                # resize_ then copy_ keeps the same buffer identity
                self._retrieval_prototypes_buf.resize_(proto_vectors.shape)
                self._retrieval_prototypes_buf.copy_(proto_vectors)
            self._retrieval_prototypes_param = None

    def _init_raw_prototypes(
        self,
        train_embeds: Tensor,  # (N,H) CPU
    ) -> Tuple[Tensor, Tensor]:
        """
        Pick prototype seeds per init_strategy.
        Returns:
          proto_indices: (P,) LongTensor of training row indices (or -1 if synthetic)
          proto_vectors: (P,H) FloatTensor in retrieval space
        """
        strategy = self.init_strategy

        if strategy == "random":
            return self._init_random(train_embeds)
        elif strategy == "random_real":
            return self._init_random_real(train_embeds)
        elif strategy == "k-medoids":
            return self._init_k_medoids(train_embeds)
        elif strategy == "k-means++":
            return self._init_k_means_pp(train_embeds)
        elif strategy == "farthest-first":
            return self._init_farthest_first(train_embeds)
        else:
            raise ValueError(f"Unknown prototype init strategy: {strategy}")

    def _init_random(
        self,
        train_embeds: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        P = self.num_prototypes
        H = self.retrieval_dim
        proto_vecs = torch.randn(P, H)
        proto_idx = torch.full((P,), fill_value=-1, dtype=torch.long)
        return proto_idx, proto_vecs

    def _init_random_real(
        self,
        train_embeds: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        N = train_embeds.size(0)
        P = self.num_prototypes
        choice = torch.randperm(N)[:P]
        proto_vecs = train_embeds[choice].clone()
        return choice.long(), proto_vecs

    def _init_k_medoids(
        self,
        train_embeds: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        P = self.num_prototypes
        embeds_cpu = train_embeds.float()  # (N,H)
        embeds_norm = _normalize(embeds_cpu)  # (N,H)
        X = embeds_norm.numpy()

        km = KMedoids(
            n_clusters=P,
            metric="euclidean",
            method="alternate",
            init="k-medoids++",
            max_iter=300,
            random_state=self.seed,
        )
        km.fit(X)

        if getattr(km, "medoid_indices_", None) is not None:
            idx = torch.as_tensor(km.medoid_indices_, dtype=torch.long)
        elif getattr(km, "medoid_indices", None) is not None:
            idx = torch.as_tensor(km.medoid_indices, dtype=torch.long)
        else:
            centers = getattr(km, "cluster_centers_", None) or getattr(
                km, "cluster_centers", None
            )
            if centers is not None:
                idx = self._nearest_point_indices(embeds_norm, centers, P)
            else:
                idx = self._farthest_first_indices(embeds_norm, P)

        # ensure uniqueness / fill shortfalls
        uniq = list(dict.fromkeys(idx.tolist()))
        if len(uniq) < P:
            extra = self._farthest_first_indices(
                embeds_norm, min(P, embeds_norm.size(0))
            ).tolist()
            pool = [i for i in extra if i not in set(uniq)]
            uniq = uniq + pool[: (P - len(uniq))]
        uniq_idx = torch.tensor(uniq[:P], dtype=torch.long)

        proto_vecs = train_embeds[uniq_idx].clone()  # (P,H)
        return uniq_idx, proto_vecs

    def _init_k_means_pp(
        self,
        train_embeds: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        P = self.num_prototypes
        embeds_cpu = train_embeds.float()  # (N,H)
        embeds_norm = _normalize(embeds_cpu)  # (N,H)
        X = embeds_norm.numpy()

        km = KMeans(
            n_clusters=P,
            init="k-means++",
            n_init=P,
            random_state=self.seed,
        )
        km.fit(X)
        centers = km.cluster_centers_  # (P,H) numpy

        idx = self._nearest_point_indices(embeds_norm, centers, P)  # (P,)
        proto_vecs = train_embeds[idx].clone()  # (P,H)
        return idx.long(), proto_vecs

    def _init_farthest_first(
        self,
        train_embeds: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        P = self.num_prototypes
        embeds_cpu = train_embeds.float()  # (N,H)
        embeds_norm = _normalize(embeds_cpu)  # (N,H)
        idx = self._farthest_first_indices(embeds_norm, P)  # (P,)
        proto_vecs = train_embeds[idx].clone()  # (P,H)
        return idx.long(), proto_vecs

    def _farthest_first_indices(
        self,
        X: Tensor,  # (N,H) normalized
        k: int,
    ) -> Tensor:
        N = X.size(0)
        if k >= N:
            return torch.arange(N, dtype=torch.long)

        g = torch.Generator().manual_seed(self.seed)
        idx0 = torch.randint(0, N, (1,), generator=g).item()

        chosen = [idx0]
        dmin = 1 - (X @ X[idx0])  # cosine distance = 1 - cos_sim
        for _ in range(1, k):
            j = torch.argmax(dmin).item()
            chosen.append(j)
            dmin = torch.minimum(dmin, 1 - (X @ X[j]))
        return torch.tensor(chosen, dtype=torch.long)

    def _nearest_point_indices(
        self,
        embeds_norm: Tensor,  # (N,H) normalized embeddings
        centers_np,  # (k,H) numpy array (normalized centers)
        k: int,
    ) -> Tensor:
        centers = torch.from_numpy(centers_np).to(embeds_norm.dtype)  # (k,H)
        dmat = torch.cdist(embeds_norm, centers, p=2)  # (N,k)

        used = set()
        out = []
        for j in range(centers.size(0)):
            col = dmat[:, j]
            _, order = torch.sort(col)
            pick = None
            for idx in order:
                ii = int(idx.item())
                if ii not in used:
                    pick = ii
                    used.add(ii)
                    break
            if pick is None:
                pick = int(order[0].item())
            out.append(pick)

        return torch.tensor(out, dtype=torch.long)

    @torch.no_grad()
    def _maybe_snap_to_real_points(
        self,
        train_embeds: Tensor,  # (N,H)
    ) -> None:
        """
        DURING TRAINING ONLY.

        Preconditions (checked by periodic_snap()):
          - trainable_prototypes == True
          - map_strategy == "projection"
          - _retrieval_prototypes_param is not None

        We:
          - find nearest neighbor in retrieval space for each prototype
          - overwrite that prototype IN-PLACE
          - update prototype_indices
        """
        if not self.trainable_prototypes:
            return
        if self.map_strategy != "projection":
            return
        if self._retrieval_prototypes_param is None:
            return

        protos_now = (
            self._retrieval_prototypes_param.data.detach().clone()
        )  # (P,H)

        # normalize both sets for cosine-ish distance via Euclidean
        train_norm = _normalize(train_embeds.float())  # (N,H)
        proto_norm = _normalize(protos_now.float())  # (P,H)

        dist_mat = torch.cdist(proto_norm, train_norm, p=2)  # (P,N)
        nn_indices = dist_mat.argmin(dim=1)  # (P,)
        snapped_vecs = train_embeds[nn_indices].clone()  # (P,H)

        # overwrite learned prototypes with nearest real examples
        self._retrieval_prototypes_param.data.copy_(snapped_vecs)

        # update prototype_indices in-place
        self.prototype_indices.data.copy_(nn_indices.long())
