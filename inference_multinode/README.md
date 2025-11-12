## Init

```
conda activate PRLE
cd /work/06782/ysu707/ls6/PRLE
```

## Viewing Logs

`tail -f $SCRATCH/prle_cache/logs//<SLURM_JOB_ID>/rank0_infer.log`


## STSB Commands

### Llama 3.3 70B Instr

```
mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/stsb/llama_33_70b_instr_icl_0.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/stsb/llama_33_70b_instr_icl_1.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/stsb/llama_33_70b_instr_icl_5.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/stsb/llama_33_70b_instr_icl_10.yaml
```

### Qwen Next 80B Instr

```
mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/stsb/qwen3_next_80b_a3b_instr_icl_0.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/stsb/qwen3_next_80b_a3b_instr_icl_1.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/stsb/qwen3_next_80b_a3b_instr_icl_5.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/stsb/qwen3_next_80b_a3b_instr_icl_10.yaml
```

## sickr_sts Commands

### Llama 3.3 70B Instr

```
mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sickr_sts/llama_33_70b_instr_icl_0.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sickr_sts/llama_33_70b_instr_icl_1.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sickr_sts/llama_33_70b_instr_icl_5.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sickr_sts/llama_33_70b_instr_icl_10.yaml
```

### Qwen Next 80B Instr

```
mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sickr_sts/qwen3_next_80b_a3b_instr_icl_0.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sickr_sts/qwen3_next_80b_a3b_instr_icl_1.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sickr_sts/qwen3_next_80b_a3b_instr_icl_5.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sickr_sts/qwen3_next_80b_a3b_instr_icl_10.yaml
```

## sts22_crosslingual Commands

### Llama 3.3 70B Instr

```
mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sts22_crosslingual/llama_33_70b_instr_icl_0.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sts22_crosslingual/llama_33_70b_instr_icl_1.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sts22_crosslingual/llama_33_70b_instr_icl_5.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sts22_crosslingual/llama_33_70b_instr_icl_10.yaml
```

### Qwen Next 80B Instr

```
mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sts22_crosslingual/qwen3_next_80b_a3b_instr_icl_0.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sts22_crosslingual/qwen3_next_80b_a3b_instr_icl_1.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sts22_crosslingual/qwen3_next_80b_a3b_instr_icl_5.yaml

mpirun -np 2 --map-by ppr:1:node --tag-output --timestamp-output \
  bash inference_multinode/multinode.sh \
  inference_multinode.infer_dm_main --config inference_multinode/configs/sts22_crosslingual/qwen3_next_80b_a3b_instr_icl_10.yaml
```