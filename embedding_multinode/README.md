# Pipeline

1. Download large model to a designated directory via `download_model.py` to
save compute space. This should be running just on a normal compute/cpu node. 

```
conda activate PRLE
cd /work/06782/ysu707/ls6/PRLE
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/scratch/06782/ysu707/prle_cache/hf
hf auth login

python -m embedding_multinode.download_model \
  --repo meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --out_dir /scratch/06782/ysu707/models/llama4_scout_17b_16e_instr

python -m embedding_multinode.download_model \
  --repo meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --out_dir /scratch/06782/ysu707/models/llama4_maverick_17b_128e_instr

python -m embedding_multinode.download_model \
  --repo meta-llama/Llama-3.3-70B-Instruct \
  --out_dir /scratch/06782/ysu707/models/llama_33_70b_instr

python -m embedding_multinode.download_model \
  --repo meta-llama/Llama-3.1-405B-Instruct \
  --out_dir /scratch/06782/ysu707/models/llama_3_405b_instr

python -m embedding_multinode.download_model \
  --repo Qwen/Qwen3-Next-80B-A3B-Instruct \
  --out_dir /scratch/06782/ysu707/models/qwen3_next_80b_a3b_instr

python -m embedding_multinode.download_model \
  --repo Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --out_dir /scratch/06782/ysu707/models/qwen3_235b_a22b_instr_2507
```

2. Because each model is likely too big to fit on a single gpu, we are sharding
the model across different gpus on different nodes. However, the issue is that 
to avoid loading the model before sharding, this requirements deepspeed to have
a manifest (mapping) that essentially tells it how to shard. 

This would require running the `make_ds_manifest.sh` file for each model of interest.
This file takes as input where the model was downloaded, and outputs a json manifest
to a specified path given the number of expected nodes and the tensor type.

```
conda activate PRLE
cd /work/06782/ysu707/ls6/PRLE

bash embedding_multinode/make_ds_manifest.sh \
    /scratch/06782/ysu707/models/qwen3_next_80b_a3b_instr \
    scratch/06782/ysu707/models/qwen3_next_80b_a3b_instr/ds_inference_tp4_bf16.json \
    -t 4 -d bf16
```

2. Compute Embeddings -- a typical workflow example may look like this:

```
idev -p gh-dev -N 8 -n 8 -t 2:00:00
idev -p gh-dev -N 2 -n 2 -t 2:00:00
idev -p gh-dev -N 4 -n 4 -t 2:00:00

idev -p gh-dev -N 2 -n 2 -t 2:00:00

cd /work/06782/ysu707/ls6/PRLE
conda activate PRLE
```

NOTE: use `tail -f $SCRATCH/prle_cache/logs/<job_id>/rank0.log` in a different 
window to view the running progress bars that do not showup correctly otherwise.

## STSB Commands

```
mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
   bash embedding_multinode/multinode.sh \
   embedding_multinode/configs/stsb/qwen3_next_80b_a3b_instr_train.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
   bash embedding_multinode/multinode.sh \
   embedding_multinode/configs/stsb/qwen3_next_80b_a3b_instr_validation_test.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
   bash embedding_multinode/multinode.sh \
   embedding_multinode/configs/stsb/llama33_70b_instr_train.yaml

tail -f $SCRATCH/prle_cache/logs/443816/rank0.log

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
   bash embedding_multinode/multinode.sh \
   embedding_multinode/configs/stsb/llama33_70b_instr_validation_test.yaml

tail -f $SCRATCH/prle_cache/logs/443891/rank0.log
```

## Sickr Commands

```
mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
   bash embedding_multinode/multinode.sh \
   embedding_multinode/configs/sickr_sts/qwen3_next_80b_a3b_instr_train.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
   bash embedding_multinode/multinode.sh \
   embedding_multinode/configs/sickr_sts/qwen3_next_80b_a3b_instr_validation_test.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
   bash embedding_multinode/multinode.sh \
   embedding_multinode/configs/sickr_sts/llama33_70b_instr_train.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
   bash embedding_multinode/multinode.sh \
   embedding_multinode/configs/sickr_sts/llama33_70b_instr_validation_test.yaml

```

## Sts crosslingual Commands

```

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
   bash embedding_multinode/multinode.sh \
   embedding_multinode/configs/sts22_crosslingual_sts/qwen3_next_80b_a3b_instr_train.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
   bash embedding_multinode/multinode.sh \
   embedding_multinode/configs/sts22_crosslingual_sts/qwen3_next_80b_a3b_instr_validation_test.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
   bash embedding_multinode/multinode.sh \
   embedding_multinode/configs/sts22_crosslingual_sts/llama33_70b_instr_train.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
   bash embedding_multinode/multinode.sh \
   embedding_multinode/configs/sts22_crosslingual_sts/llama33_70b_instr_validation_test.yaml

```
