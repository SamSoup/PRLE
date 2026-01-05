

```
python generate_value_space_embeddings.py \
  --config config/stsb/PRLE_v2/BEST_llama_3.1_8B_Instr.yaml \
  --value_cache_dir /scratch/06782/ysu707/PRLE/stsb/llama_3.1_8B_Instr/kernel

python generate_value_space_embeddings.py \
  --config config/stsb/PRLE_v2/BEST_llama_3.3_70B_Instr.yaml \
  --value_cache_dir /scratch/06782/ysu707/PRLE/stsb/llama_33_70b_instr/kernel

python generate_value_space_embeddings.py \
  --config config/sickr_sts/PRLE_v2/BEST_llama_3.1_8B_Instr.yaml \
  --value_cache_dir /scratch/06782/ysu707/PRLE/sickr_sts/llama_3.1_8B_Instr/kernel

python generate_value_space_embeddings.py \
  --config config/sickr_sts/PRLE_v2/BEST_llama_3.3_70B_Instr.yaml \
  --value_cache_dir /scratch/06782/ysu707/PRLE/sickr_sts/llama_33_70b_instr/kernel
```


```
python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/stsb/llama_3.1_8B_Instr/kernel \
  --dataset_name sentence-transformers/stsb \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/stsb/llama_3.1_8B_Instr/wrapperbox/kernel

python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/stsb/llama_33_70b_instr/kernel \
  --dataset_name sentence-transformers/stsb \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/stsb/llama_33_70b_instr/wrapperbox/kernel

python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/sickr_sts/llama_3.1_8B_Instr/kernel \
  --dataset_name Samsoup/sickr-sts \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/sickr_sts/llama_3.1_8B_Instr/wrapperbox/kernel

python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/sickr_sts/llama_33_70b_instr/kernel \
  --dataset_name Samsoup/sickr-sts \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/sickr_sts/llama_33_70b_instr/wrapperbox/kernel
```