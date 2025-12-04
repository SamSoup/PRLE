# Prototypical Regression with Local Experts

## High-level Goal 

We're building an interpretable, prototype-driven regression model for text inputs. 

**Task setting:** 

Input: text (NLP datasets, e.g. STS-B sentence pairs, toxicity ratings, etc.) 
Output: scalar regression target (similarity score, toxicity score, etc.). 

**Core motivation:** Traditional fine-tuned regressors (e.g. frozen BERT + linear head, or full finetune) get strong performance but are hard to interpret. KNN and decision trees are more interpretable but often weaker. We want to beat those baselines (or come close) while giving: 

**Local experts**: different small regressors specialized to different regions of data. 
**Case-based attribution**: “this prediction was mostly influenced by these specific training examples.”
**Prototype geometry**: structure in embedding space that makes sense with respect to labels. 

We aim to get: 

1. Accuracy close to (or better than) a strong frozen-encoder linear regressor baseline. 
2. Interpretability via explicit prototype examples. 
3. Efficiency at inference (much cheaper than literal kNN over all training points). 

## Conceptual Architecture 

The full model has four conceptual blocks: 

### 0. Frozen encoder → cached embeddings 

We **never** backprop through the text encoder during PRLE training. 

We run each split (train/val/test) through a chosen encoder once (e.g. SBERT, Llama embedding adapter, mean-pooled HF model), get a fixed dense vector per example. 

We cache those embeddings to disk (.npy) and load them fast in training. This gives us: 
    * train_embeddings: shape (N_train, H) 
    * train_labels: shape (N_train,) (and similarly for val/test) 

This is handled by EmbeddingDataModule. 

### 1. PrototypeManager (geometry + interpretability) 

This manages “prototypes,” i.e. representative points in embedding space. 

**Retrieval space** We choose num_prototypes = P vectors in the same space as the frozen embeddings. 
We select them using a configurable init_strategy, e.g.: 
* random_real: sample P real training points
* kmeans++: KMeans++, take centroid→nearest real point
* k-medoids: medoids from clustering. 
* farthest_first: greedy max-min cover. 
* Optionally, prototypes are learnable (they can move), but after each epoch we can **snap them back** to the nearest real training example (map_strategy: projection). This preserves interpretability: each prototype corresponds to “this specific training point X.” So retrieval space = where routing/nearest-neighbor happens. 

**Value space (optional)** We optionally learn a projection head f(·) that maps embeddings from retrieval space (dim H) to a new “value space” (dim proj_dim). This value space is optimized with metric learning: Points with similar labels are pulled closer. Points with very different labels are pushed apart. The same projection is applied to all prototypes; this is the space that experts operate in and make predictions from. If `use_value_space = false`, then value space = retrieval space and projection is effectively identity/frozen. 

So: **Retrieval space decides “who talks.”** **Value space encodes “what they say.”** 

### 2. ActivationRouter (routing / gating / mixture weights) 

Given: Input embedding in retrieval space, Prototypes in retrieval space, Same input + prototypes in value space, this module decides which prototypes are active and how much they matter. We support multiple routing strategies: 
* knn: Activate only top-k closest prototypes in retrieval space. Normalize their influence with a softmax over negative distance. Gives sparse, very interpretable assignments. 
* radius: Activate any prototypes within a distance threshold in retrieval space. If none qualify, fall back to the nearest one. Good for local region modeling. 
* softmax: Dense, differentiable: weights = softmax(-τ * dist). `τ` is a learned temperature. Learns how sharp the assignment should be. Fully end-to-end with gradients. 
* mlp_sparse: Learn a small MLP that scores each (input, prototype) pair in **value space** using features like [x_val, p_val, |x_val - p_val|, ||x_val - p_val||]. Use Gumbel-softmax to encourage sparse routing. This allows routing to ignore “nearest in raw embedding” and instead learn semantic specialization in task space. All routers present a unified API:

```python
weights = router(
    retrieval_embeds,   # (B,H_retr)
    retrieval_protos,   # (P,H_retr)
    value_embeds,       # (B,D_val)
    value_protos,       # (P,D_val)
) # returns (B,P), each row sums to 1
```

Routers can have trainable parameters: e.g. softmax has learnable τ, mlp_sparse has an MLP. We can freeze or unfreeze them phase-by-phase (see EM training below). 

### 3. Local Experts Instead of one global regressor, we learn *one expert per prototype*. 

Current implemented expert class: 

**LinearExpertPRLE**: ModuleList of P small linear regressors, each Linear(D_val, 1). Each expert sees the value-space embedding of the input (not raw text). Each expert predicts a scalar output. At inference: 
1. For each input x, compute value embedding x_val. 
2. Run all experts to get predictions expert_outputs with shape (B,P) (or (B,P,1)). 
3. Get activation weights (B,P) from the ActivationRouter. 
4. Combine: final prediction = sum over prototypes of weight[p] * expert_outputs[p]. 

Interpretation: "These 3 prototypes contributed 80% of the score, here are their training examples and their local expert predictions." 

Initialization of experts: We’ve set the design for: 
* **random**: random linear weights per expert. 
* **meta linear regression** (planned): fit a single global linear regression on the frozen embeddings to predict labels, then initialize each expert with those weights (so they start reasonable, not random). 
* **meta classification head** (planned): initialize each expert from a pretrained classification/regression head (e.g. a pretrained BERT linear head) adapted to MSE. The per-prototype experts are always considered trainable in the "experts" phase. 

Training Procedure: 

Data flow: 
* EmbeddingDataModule: Produces batches like {"embeddings": (B,H), "labels": (B,)} instead of raw text. Holds train_embeddings and train_labels tensors so the model can: initialize prototypes from training data, compute prototype self-consistency loss, perform periodic snapping. 

Model forward (BasePrototypicalRegressor): 
1. Ensure prototypes are initialized (first time we see data). 
2. Get: * retrieval_protos (P,H) * value_protos (P,D_val) 
3. Project batch embeddings to value space → value_embeds (B,D_val). 
4. Router gives gating weights weights (B,P). 
5. Experts predict per-prototype outputs → expert_matrix (B,P). 
6. Weighted sum → final preds (B,). It also returns intermediates needed for losses. 

Losses:
We train multiple objectives, which have different roles: 
1. **Task loss (L_task)** * Standard MSE between final prediction and ground truth label. This is what you'd optimize if you were just training one regression head. 
2. **Metric learning loss (L_metric)** * Contrastive-style loss in value space. Encourages embeddings with similar labels to be close, dissimilar labels to be far. Shapes the learned projection head and (if allowed) prototype positions, so geometry actually reflects label structure, not just encoder similarity. Only makes sense if `use_value_space = true`. 
3. **Prototype self-fit loss (L_proto_self)** Each prototype maps to (or snaps to) a specific training example (the "anchor"). The prototype’s *own* expert should predict that anchor’s true label. This aligns the local expert with its exemplar. Interpretability story: “Prototype p stands for training point i, and expert p reproduces label(y_i).” 
4. **Local responsibility loss (L_local)** For each data point x, look at *all* expert predictions, compare each expert’s output to the true label of x, weight by router weights. This encourages each expert to fit the region of data it is actually responsible for. 
5. **Consistency loss (L_consistency)** * If a prototype gets high weight for input x, their value-space embeddings shouldn’t be far apart. Penalizes routing that assigns influence from prototypes that are very distant in value space. Intuition: "Don't let distant prototypes pretend to speak for this example." We then take a weighted sum of these terms. 

EM-style Alternating Training: 
Because there are multiple moving parts (prototypes + projection head geometry, and experts + routing), we support EM-like alternating phases across epochs: 
* **geometry phase** * Train PrototypeManager parts: projection head (value space), prototypes (if trainable), metric learning, Freeze experts and router, Emphasize metric shaping + snapping + consistency.
* **experts phase** * Freeze geometry (prototypes fixed, projection head frozen). Train experts + router/gating. Emphasize local fit, prototype self-fit, consistency  
* **all_unfrozen mode (no EM)** Everything trains together; loss weights are just averaged. You control this with: `train.em_alt_training: true` in the yaml config.

At the end of each epoch we also call PrototypeManager.periodic_snap(...) to re-align any drifting prototypes with the nearest real training example so we keep the “this prototype == that training example” interpretability story. --- 

## Repo Structure
.
├── configs/
│   ├── stsb
│   |   ├── PRLE
|   │   |   ├── llama_3.1_8B_Instr.yaml
│   └── ... (other dataset configs)
│
├── data/
│   ├── base.py
│   ├── stsb.py                 # original text DataModule (raw text or tokenized)
│   ├── ...                     # Other datasets
│   ├── embedding.py            # EmbeddingDataModule: caches encoder outputs to .npy and serves them
│   └── factory.py              # get_datamodule(name=...)
│
├── model/
│   ├── base.py                 # BasePrototypicalRegressor (LightningModule, losses, EM control)
│   ├── linear.py               # LinearExpertPRLE (per-prototype linear regressors as experts)
│   ├── factory.py              # get_model(model_type=...) -> returns LinearExpertPRLE, etc.
│   │
│   ├── prototypes/
│   │   ├── __init__.py         # exports PrototypeManager, build_prototype_manager
|   |   ├── projection_heads.py # Implemented various projection kernels for mapping retreival to value space -- LinearProjectionHead, MLPProjectionHead, RBFRandomFourierProjectionHead
│   │   └── base.py             # PrototypeManager class:
│   │                           #   - stores prototypes
│   │                           #   - projection head (value space)
│   │                           #   - metric learning loss
│   │                           #   - init strategies (random_real, kmeans++, etc.)
│   │                           #   - snapping logic
│   │
│   ├── activations.py          # ActivationRouter + implementations:
│   │                           #   - KNNRouter
│   │                           #   - RadiusRouter
│   │                           #   - SoftmaxRouter (learnable τ)
│   │                           #   - SparseMLPRouter (gumbel-softmax gating MLP)
│   │
│   └── encoders/
│       ├── factory.py          # get_encoder(encoder_type=..., model_name=...)
│       └── ...                 # sentence encoder wrapper, mean-pool encoder, etc.
│
├── train.py                    # main training script
├── requirements.txt            
└── README.md