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
    "projection",  # DURING TRAINING: periodically snap drifting trainable prototypes
    # back to nearest real examples
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
    PrototypeManager is responsible for:
      - The prototype matrix in RETRIEVAL space, shape (P, H_retrieval).
        This matrix can be:
           * trainable (nn.Parameter) OR
           * frozen (buffer)
        and we keep an index mapping to real training rows for interpretability.

      - The VALUE space transform g(·):
           If use_value_space=True:
               g is a trainable Linear: R^{H_retrieval} -> R^{proj_dim}
               and we apply metric learning loss here.
           If use_value_space=False:
               g is effectively identity; proj_dim == retrieval_dim;
               g is frozen; metric learning loss should be 0.

      - Optional periodic snapping DURING TRAINING ONLY:
         If (trainable_prototypes == True AND map_strategy == "projection"),
         we can "snap" trainable prototype vectors back to nearest real training
         embeddings every epoch to preserve interpretability.

      - Exposing knobs to freeze/unfreeze:
          * prototype params
          * projection head params
        so the LightningModule can do EM-style alternating optimization between
        geometry (g/prototypes) and experts.

    IMPORTANT:
      We NEVER snap at init. The init strategy alone defines interpretability
      at birth. Snapping is only a periodic training-time tether.
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

        # if we are NOT using a separate value space, force proj_dim = retrieval_dim
        self.proj_dim = proj_dim if (proj_dim is not None) else retrieval_dim

        # normalize init strategy aliases (kmeans++ vs k-means++, etc.)
        self.init_strategy = _normalize_strategy(init_strategy)
        self.map_strategy = map_strategy  # only applies during training
        self.distance_metric = distance_metric
        self.trainable_prototypes = bool(trainable_prototypes)
        self.seed = seed

        # After init() completes:
        # - If trainable: _retrieval_prototypes_param is nn.Parameter
        # - Else:         we'll later create a persistent buffer _retrieval_prototypes_buf
        self._retrieval_prototypes_param: Optional[nn.Parameter] = None
        # NOTE: do NOT assign self._retrieval_prototypes_buf here.
        # We'll create it via register_buffer(...) later if needed.
        # This avoids KeyError when calling register_buffer.

        # prototype_indices: (P,)
        # ties each prototype to a training row index (or -1 if synthetic)
        # We'll register as a buffer so it survives checkpoints.
        self.register_buffer(
            "prototype_indices",
            torch.full(
                (self.num_prototypes,),
                fill_value=-1,
                dtype=torch.long,
            ),
            persistent=True,
        )

        # Track whether we already initialized prototypes from train data.
        # Also register a tiny buffer flag for checkpoint restore.
        self._is_initialized = False
        self.register_buffer(
            "_is_initialized_buf",
            torch.tensor(0, dtype=torch.uint8),
            persistent=True,
        )

        # --- projection head g(·) ---
        # If use_value_space:
        #   train a Linear head that maps retrieval -> value space, and expose
        #   metric_learning_loss() to shape this geometry.
        # Else:
        #   g is effectively identity, and we will NOT train it.
        if self.use_value_space:
            self.projection_head: nn.Module = nn.Linear(
                self.retrieval_dim,
                self.proj_dim,
                bias=False,
            )
            self._projection_trainable = True  # default, can be toggled for EM
        else:
            # Identity mapping, no params, always frozen
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
        One-time pre-training setup.

        1) Run the chosen init strategy to get:
           proto_indices: (P,) LongTensor
           proto_vectors: (P,H_retrieval) FloatTensor
           (No snapping here. We trust the init strategy.)

        2) Register those vectors as either trainable param or frozen buffer.

        3) Save indices for interpretability.
        """
        if self._is_initialized:
            return

        proto_indices, proto_vectors = self._init_raw_prototypes(train_embeds)
        # proto_indices: (P,)
        # proto_vectors: (P,H)

        self._register_retrieval_prototypes(proto_vectors)

        # update prototype_indices buffer in-place
        self.prototype_indices.data.copy_(proto_indices.long())

        self._is_initialized = True
        self._is_initialized_buf.data.copy_(torch.tensor(1, dtype=torch.uint8))

    def periodic_snap(
        self,
        train_embeds: Tensor,  # (N,H)
    ) -> None:
        """
        Training-time interpretability tether.

        If:
          - prototypes are trainable,
          - and map_strategy == "projection",
        then at the END OF EPOCH we:
          - for each learned prototype vector,
            find the nearest frozen training embedding in retrieval space,
          - overwrite that prototype row IN-PLACE,
          - update prototype_indices.

        If prototypes are frozen (buffer mode), or map_strategy == "none",
        this is a no-op.
        """
        if not self.trainable_prototypes:
            return
        if self.map_strategy != "projection":
            return
        if self._retrieval_prototypes_param is None:
            return

        self._maybe_snap_to_real_points(train_embeds)

    def get_retrieval_prototypes(self) -> Tensor:
        """
        Return retrieval-space prototypes (P,H) on the same device
        as projection_head (so downstream code can stay simple).
        """
        dev = (
            next(
                self.projection_head.parameters(),
                torch.tensor([], device="cpu"),
            ).device
            if hasattr(self.projection_head, "parameters")
            else torch.device("cpu")
        )

        if self.trainable_prototypes:
            assert self._retrieval_prototypes_param is not None
            return self._retrieval_prototypes_param.to(dev)
        else:
            # buffer version
            assert hasattr(self, "_retrieval_prototypes_buf")
            return getattr(self, "_retrieval_prototypes_buf").to(dev)

    def project_embeddings(self, x: Tensor) -> Tensor:
        """
        Map retrieval embeddings -> VALUE space.

        If use_value_space=True, this is a learned linear projection (trainable
        depending on current EM phase). If use_value_space=False, this is
        effectively identity and frozen.
        """
        return self.projection_head(x)

    def get_value_prototypes(self) -> Tensor:
        """
        VALUE-space prototypes g(proto). We detach for inspection.
        """
        with torch.no_grad():
            rp = self.get_retrieval_prototypes()
            vp = self.project_embeddings(rp)
        return vp

    # ------------------------------------------------------------------
    # EM-style phase control knobs
    # ------------------------------------------------------------------

    def enable_projection_training(self, flag: bool) -> None:
        """
        Turn gradient updates on/off for the projection head g(·).
        Only meaningful if use_value_space=True.

        flag=True  -> g.train(), requires_grad=True
        flag=False -> g.eval(),  requires_grad=False

        NOTE: This does NOT affect experts or prototypes, it's only the projection head.
        """
        if not self.use_value_space:
            # identity head, no params anyway
            self._projection_trainable = False
            return

        self._projection_trainable = bool(flag)
        self.projection_head.train(flag)
        for p in self.projection_head.parameters():
            p.requires_grad = flag

    def enable_prototype_training(self, flag: bool) -> None:
        """
        Toggle whether prototypes themselves (the retrieval vectors) get updated
        by gradient descent in this phase.

        Only relevant if trainable_prototypes=True (i.e. they are nn.Parameter).
        Frozen-buffer prototypes ignore this.

        NOTE: This does NOT handle snapping; snapping is still periodic_snap().
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
        Shape the VALUE space so that similar labels -> close, dissimilar labels -> far.

        If use_value_space=False, or projection head is frozen for this EM phase,
        we just return 0.0 so we don't fight experts in phases where geometry
        is supposed to be fixed.
        """
        # If we aren't even *using* a distinct value space, skip
        if not self.use_value_space:
            return torch.zeros((), device=value_embeds.device)

        # If projection head is present but we're currently freezing it,
        # then don't drive gradients through it right now.
        if not self._projection_trainable:
            return torch.zeros((), device=value_embeds.device)

        if value_embeds.size(0) < 2:
            return torch.zeros((), device=value_embeds.device)

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
    # Optional: prototype pruning / replacement (future)
    # ------------------------------------------------------------------

    def replace_or_prune_prototypes(
        self,
        train_embeds: Tensor,
        train_labels: Optional[Tensor] = None,
        usage_stats: Optional[Tensor] = None,
    ) -> None:
        """
        Future work only.
        """
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
        a trainable nn.Parameter or a frozen buffer.
        """
        proto_vectors = proto_vectors.float()

        if self.trainable_prototypes:
            # trainable => keep as nn.Parameter
            self._retrieval_prototypes_param = nn.Parameter(proto_vectors)
        else:
            # frozen => persistent buffer
            self._retrieval_prototypes_param = None
            # now safely register buffer because we did NOT predefine
            # _retrieval_prototypes_buf in __init__
            self.register_buffer(
                "_retrieval_prototypes_buf",
                proto_vectors,
                persistent=True,
            )

    def _init_raw_prototypes(
        self,
        train_embeds: Tensor,  # (N,H) CPU
    ) -> Tuple[Tensor, Tensor]:
        """
        Pick prototype seeds per init_strategy.
        Returns:
          proto_indices: (P,) LongTensor  (row indices or -1 if synthetic)
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
        Internal: DURING TRAINING ONLY.

        Preconditions (enforced by periodic_snap()):
          - trainable_prototypes == True
          - map_strategy == "projection"
          - _retrieval_prototypes_param is not None

        We compute nearest neighbor in retrieval space for each prototype
        and overwrite that prototype IN-PLACE, then update prototype_indices.
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

        train_norm = _normalize(train_embeds.float())  # (N,H)
        proto_norm = _normalize(protos_now.float())  # (P,H)

        dist_mat = torch.cdist(proto_norm, train_norm, p=2)  # (P,N)
        nn_indices = dist_mat.argmin(dim=1)  # (P,)
        snapped_vecs = train_embeds[nn_indices].clone()  # (P,H)

        # overwrite learned prototypes with nearest real examples
        self._retrieval_prototypes_param.data.copy_(snapped_vecs)

        # update buffer prototype_indices in-place
        self.prototype_indices.data.copy_(nn_indices.long())
