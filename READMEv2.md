# Prototypical Regression with Local Experts (PRLE v2)

## High-level Goal

We’re building an interpretable, prototype-driven regression model for text inputs.

**Task setting**

- **Input:** text (e.g., STS-B sentence pairs, toxicity ratings, etc.)  
- **Output:** scalar regression target (similarity score, toxicity score, etc.)

**Core motivation**

Traditional frozen-encoder regressors (e.g., encoder + linear head, SVR on embeddings) get strong performance but are hard to interpret. KNN / trees are somewhat interpretable but weaker or expensive at inference.

We want to approach or beat strong frozen-encoder baselines while providing:

- **Local experts:** different small regressors specialized to different regions of the space.
- **Case-based attribution:** “this prediction was mostly influenced by these specific training examples.”
- **Prototype geometry:** explicit, human-inspectable structure in embedding space that tracks labels.

---

## What’s new in v2 (vs v1)

PRLE v2 adds a **metric-learning kernel stage** and tight tooling to understand whether “local experts” actually help.

### 1. Two-stage pipeline: kernel → experts

We now explicitly split the pipeline:

1. **Stage 0: Frozen encoder → retrieval space**
   - Frozen text encoder (e.g. `meta-llama/Llama-3.1-8B-Instruct`) produces a fixed embedding for each example.
   - Embeddings are cached via `EmbeddingDataModule` to disk (`train_embeds.npy`, etc.).

2. **Stage 1: Kernel metric learner → value space**
   - A `KernelMetricLearner` projects retrieval embeddings into a **value space** via a projection head (e.g., ResMLP).
   - Trained with metric learning so that **distance in value space correlates with label differences**.
   - In practice on STS-B + Llama 3.1 8B:
     - Retrieval space: `corr(|Δy|, dist) ≈ 0`
     - Value space: `corr(|Δy|, dist) ≈ 0.93`  
       → value space is strongly label-shaped.

3. **Stage 2: Prototypes + router + experts (PRLE proper)**
   - Prototypes live in (and route in) this learned value space.
   - Local experts operate over value embeddings, using routing weights derived from distances / learned gating.

The kernel stage is shared across any downstream expert architecture; it’s a reusable “learned regression metric.”

### 2. Stronger empirical picture

On **STS-B with Llama 3.1 8B embeddings**, we now have:

- **Prompt baselines** (zero/one/few-shot) as a reference.
- **Wrapper baselines** (Linear, KNN, SVR, DecisionTree, RandomForest) in:
  - **Retrieval space** (raw encoder embeddings).
  - **Value space** (after kernel projection).

Findings:

- **Linear regression**:
  - Retrieval space: Pearson ≈ 0.585
  - Value space: Pearson ≈ 0.651  
  → kernel clearly makes the space more regression-friendly.

- **Non-linear baselines in value space**:
  - KNN, SVR, Decision Trees, Random Forests all cluster around:
    - MSE ≈ 0.051–0.056
    - Pearson ≈ 0.689–0.693
  - KNN ≈ SVR in value space.

- **Local-expert / MoE sanity check (oracle test)**:
  - Cluster value-space embeddings with K-medoids (P = 25).
  - For each cluster, train an independent regressor (SVR or Linear) on that cluster only.
  - At test time, assign each test point to nearest medoid and use that cluster’s regressor.
  - **Result:** Oracle MoE does **not** beat a strong global SVR / KNN in value space.

**Conclusion so far (for STS-B + Llama 3.1 8B):**

- The **value space kernel is doing exactly what we want geometrically** (distance ≈ label difference).
- This already makes **simple global models** (Linear, KNN, SVR) quite strong.
- The value space behaves like a **single smooth manifold**; splitting into many local experts brings little extra signal for this particular dataset+encoder.

PRLE v2 is therefore best viewed as:

> A **label-aware metric + prototype layer** that can support interpretable modeling, which may or may not need local experts depending on the dataset.

---

## Conceptual Architecture (v2)

We now describe the components in the v2 pipeline.

### 0. Frozen encoder → cached embeddings (retrieval space)

We **never** backprop through the text encoder during PRLE training.

1. Wrap the dataset with a raw DataModule (e.g. `STSBDataModule`).
2. Use `EmbeddingDataModule` with a chosen encoder (sentence encoder, mean-pool HF encoder, etc.).
3. On `setup()`, we:
   - Run the encoder over train/validation/test once.
   - Cache embeddings to `{embedding_cache_dir}/{split}_embeds.npy`.
   - Reload labels from the original HF dataset each run (labels are *not* cached to disk).

Outputs in memory:

- `train_embeddings`: `(N_train, H)`
- `val_embeddings`: `(N_val, H)`
- `test_embeddings`: `(N_test, H)`
- Corresponding label tensors.

This is the **retrieval space**: the raw encoder embedding space.

---

### 1. KernelMetricLearner → value space

The kernel is a Lightning module (`KernelMetricLearner`) trained in Stage 1.

- Uses a projection head from `model/prototypes/projection_heads.py`:
  - e.g. `ResMLPProjectionHead` with configurable depth, width, activation.
- Maps retrieval space → value space:

  ```python
  z = projection_head(e)  # e: retrieval embedding, z: value embedding
  z = normalize(z)
  ```

* Optimized with a metric-learning loss over pairs/triplets:

  * Main objective: **pairwise similarity regression** so that distances in value space track label differences.
  * Optional uniformity / regularization terms.

At the end of Stage 1, we:

* Save a checkpoint in `kernel.checkpoint_dir` (see configs).
* During Stage 2, we **freeze this projection head** and reuse it for both data and prototypes.

The **value space** is thus a label-aligned embedding space shared by all downstream models.

---

### 2. PrototypeManager (prototypes in retrieval + value space)

`PrototypeManager` is responsible for:

* Storing prototype indices and vectors.
* Running the projection head to get value-space prototypes.
* Prototype initialization and optional snapping to real training examples.

Key ideas:

* Prototypes are **real points** (via indices into `train_embeddings`), not arbitrary vectors.
* Initialization strategies (v2):

  * `k-medoids` (default for v2)
  * `k-means++`
  * `farthest_first`
* In v2, for many configs:

  * `trainable_prototypes = false`
  * `map_strategy = none`
    → prototypes are fixed once initialized, simplifying interpretation: prototype p corresponds to a concrete training example.

We can work in:

* **Retrieval space:** raw encoder embeddings (e.g. for some routers).
* **Value space:** metric-learned space (for routing and experts).

The same prototype indices are used in both spaces by projecting their retrieval embeddings through the kernel.

---

### 3. ActivationRouter (routing / gating)

Given:

* Batch embeddings in retrieval space: `(B, H_retr)`
* Prototypes in retrieval space: `(P, H_retr)`
* Corresponding value-space embeddings for batch + prototypes: `(B, D_val)`, `(P, D_val)`

The router outputs **mixture weights** over prototypes:

```python
weights = router(
    retrieval_embeds,   # (B, H_retr)
    retrieval_protos,   # (P, H_retr)
    value_embeds,       # (B, D_val)
    value_protos,       # (P, D_val)
)  # (B, P), rows sum to 1
```

Implemented routers:

* **KNNRouter**

  * k-NN in retrieval space, softmax over negative distance among top-k.
* **RadiusRouter**

  * Activate any prototypes within a distance threshold, fallback to nearest if empty.
* **SoftmaxRouter (default in many v2 configs)**

  * Dense, differentiable: `weights = softmax(-τ * dist)`.
  * τ is learnable and can be annealed over training.
* **SparseMLPRouter**

  * MLP over value-space features such as `[x_val, p_val, |x_val - p_val|, ||x_val - p_val||]`.
  * Uses Gumbel-softmax / sparsity encouragement to get few active prototypes.

Routers can be frozen or trained depending on the stage.

---

### 4. Local Experts

Instead of a single global regressor, we have *per-prototype experts*.

Current implemented expert class:

* **LinearExpertPRLE**

  * A `ModuleList` of P linear maps: `Linear(D_val, 1)` each.
  * Each expert sees the **value-space embedding** of the input.
  * Forward pass:

    1. Compute value embedding `z` from retrieval embedding.
    2. Compute all expert outputs: `expert_outputs` of shape `(B, P)`.
    3. Get weights `weights` from the router: `(B, P)`.
    4. Final prediction:

       ```python
       y_hat = (weights * expert_outputs).sum(dim=-1)  # (B,)
       ```

Interpretation at inference time:

* “These k prototypes contributed most; here are their training sentences and their local predictions.”

Expert initialization strategies currently implemented or planned:

* **random**: default, per-expert linear initialized randomly.
* **global-linear init (planned)**:

  * Fit a global linear regressor in value (or retrieval) space.
  * Initialize each expert with the same global weights; training then lets them specialize.

---

### 5. Losses (v2 formulation)

In v2, we consolidate losses around the notion of:

> global task performance + local expert fitness + good routing behavior.

v2 loss terms (see `loss` section in configs):

1. **Task main loss (`lambda_task_main`)**

   * Standard MSE between final prediction and ground-truth label.
   * What you’d optimize for a single global regressor.

2. **Expert fit loss (`lambda_expert_fit`)**

   * Responsibility-weighted expert MSE.
   * Encourages each expert to be accurate on examples it’s responsible for.

3. **Anchor routing loss (`lambda_anchor_route`)**

   * Aligns routing behavior with prototype anchors (their “home” region).

4. **Balance loss (`lambda_balance`)**

   * Load-balancing term for routers with trainable parameters (e.g., `mlp_sparse`).
   * Prevents degenerate behavior where one prototype explains everything.

5. **Entropy loss (`lambda_entropy`)**

   * Optional; can sharpen or smooth routing by penalizing high/low entropy over prototypes.

The older v1 loss terms (geometry, proto_self, local, consistency) are still present for backward compatibility but are set to zero when `use_v2_loss = true`.

---

### 6. Training pipeline & stages

Training is driven by a **YAML config** under `configs/`.

Key sections:

* `data`: dataset name, max seq length, embedding cache dir, combine_fields, etc.
* `model`: encoder type/name, hidden dim, `num_prototypes`, distance metric, init strategy, whether to use value space, routing strategy, etc.
* `kernel`: kernel stage hyperparameters (projection type, metric losses, learning rate, etc.).
* `train`: learning rate, batch sizes, early stopping, etc.
* `loss`: v2 lambda weights.
* `pipeline.stages`: which stages to run, e.g. `["kernel", "experts"]`.

Two-stage flow:

1. **Kernel stage**

   * Train `KernelMetricLearner` in isolation using retrieval embeddings + labels.
   * Save checkpoint to `kernel.checkpoint_dir`.
2. **Experts stage**

   * Freeze kernel projection head.
   * Construct `BasePrototypicalRegressor` / `LinearExpertPRLE` using value-space from the frozen kernel.
   * Initialize prototypes (e.g., k-medoids in value space).
   * Train experts + router with v2 loss.

---

## Analysis & Debugging Tools

To understand and validate v2, we include standalone scripts (examples):

* **Retrieval vs value space analysis**

  * Compute embeddings via `EmbeddingDataModule`.
  * Load kernel checkpoint.
  * Project to value space.
  * Run:

    * SVR and Linear Regression **before vs after** kernel.
    * Distance–label correlation (|Δy| vs distances).
    * t-SNE plots of retrieval vs value space with train/val/test + prototypes.

* **Oracle MoE in value space**

  * Run K-medoids clustering in value space.
  * Train independent regressors per cluster.
  * Use nearest medoid routing at test time.
  * Compare to global SVR / KNN to answer:

    > “Does local specialization even help on this dataset+encoder+kernel?”

* **Wrapper baselines in both spaces**

  * Helper scripts to generate value-space embeddings from retrieval cache using the kernel checkpoint.
  * Reuse `run_wrapperbox.py` to benchmark Linear, KNN, SVR, DecisionTree, RandomForest in both retrieval and value spaces.

These tools are critical to diagnose whether PRLE is underperforming because:

* the metric space is bad, or
* the experts/router architecture is not exploiting a good metric, or
* there is simply **no extra headroom** beyond a smooth global regressor.

---

## Repo Structure (v2)

```text
.
├── configs/
│   ├── stsb/
│   │   └── PRLE/
│   │       └── llama_3.1_8B_Instr.yaml     # example STS-B + Llama 3.1 8B config
│   └── ...                                 # other datasets / models
│
├── data/
│   ├── base.py                             # base DataModule abstractions
│   ├── stsb.py                             # STS-B DataModule (raw text or tokenized)
│   ├── ...                                 # other datasets
│   ├── embedding.py                        # EmbeddingDataModule: caches encoder outputs to .npy
│   └── factory.py                          # get_datamodule(name=...)
│
├── model/
│   ├── base.py                             # BasePrototypicalRegressor (LightningModule, v2 losses)
│   ├── linear.py                           # LinearExpertPRLE (one Linear head per prototype)
│   ├── factory.py                          # get_model(model.type=...) -> LinearExpertPRLE, etc.
│   │
│   ├── kernel.py                           # KernelMetricLearner: Stage-1 metric-learning model
│   │
│   ├── prototypes/
│   │   ├── __init__.py                     # exports PrototypeManager, projection heads
│   │   ├── projection_heads.py             # Linear, MLP, ResMLP, RFF-based heads
│   │   └── base.py                         # PrototypeManager:
│   │                                       #   - stores prototype indices/vectors
│   │                                       #   - interfaces with projection head
│   │                                       #   - init strategies (k-medoids, k-means++, farthest_first, ...)
│   │                                       #   - snapping / mapping logic
│   │
│   ├── activations.py                      # ActivationRouter implementations:
│   │   #   - KNNRouter
│   │   #   - RadiusRouter
│   │   #   - SoftmaxRouter (learnable τ)
│   │   #   - SparseMLPRouter (MLP-based sparse gating)
│   │
│   └── encoders/
│       ├── factory.py                      # get_encoder(encoder_type=..., model_name=...)
│       └── ...                             # sentence encoders, mean-pool HF wrappers, etc.
│
├── analysis/
│   ├── run_wrapperbox.py                   # classical regression baselines on cached embeddings
│   ├── generate_value_space_embeds.py      # apply kernel to retrieval cache
│   ├── analyze_value_vs_retrieval.py       # retrieval vs value space analysis + t-SNE
│   ├── oracle_moe_value_space.py           # K-medoids + per-cluster regressor sanity check
│   └── ...                                 # other exploration scripts
│
├── train.py                                # main training entrypoint (orchestration of stages)
├── requirements.txt
└── README.md                               # this file
```

---

## Current Takeaways (STS-B + Llama 3.1 8B)

* The **value-space kernel is validated**: it produces label-aware geometry.
* This geometry **significantly helps simple global models** (e.g. Linear).
* Non-linear baselines (KNN, SVR, RF, DT) converge to similar performance in value space.
* **Oracle MoE** (perfect clustering + per-cluster regressors) does *not* outperform global SVR/KNN → limited benefit from local experts for this dataset.
* PRLE v2, as currently instantiated, is **competitive but not yet surpassing** the strongest wrapper-based baselines here.

Future work in this repo will focus on:

* Metric-aware prototype regressors (e.g. prototype-based kernel regression),
* Distilling strong global models into small, interpretable prototype models,
* Evaluating on datasets where local structure is richer and MoE has more opportunity to shine.
