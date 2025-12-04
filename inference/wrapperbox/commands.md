# Commands 

Setup:

```
conda activate PRLE
cd /work/06782/ysu707/ls6/PRLE/inference/wrapperbox
```

## STS-B

```
python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/stsb/llama_3.1_8B_Instr \
  --dataset_name sentence-transformers/stsb \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/stsb/llama_3.1_8B_Instr/wrapperbox

python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/stsb/llama_33_70b_instr \
  --dataset_name sentence-transformers/stsb \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/stsb/llama_33_70b_instr/wrapperbox
```

## SICKR_STS

```
python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/sickr_sts/llama_3.1_8B_Instr \
  --dataset_name Samsoup/sickr-sts \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/sickr_sts/llama_3.1_8B_Instr/wrapperbox
&& \
python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/sickr_sts/llama_33_70b_instr \
  --dataset_name Samsoup/sickr-sts \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/sickr_sts/llama_33_70b_instr/wrapperbox

```

## STS22_CROSSLINGUAL

```

python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/sts22/llama_3.1_8B_Instr \
  --dataset_name Samsoup/sts22-crosslingual-sts \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/sts22/llama_3.1_8B_Instr/wrapperbox
&& \
python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/sts22/llama_33_70b_instr \
  --dataset_name Samsoup/sts22-crosslingual-sts \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/sts22/llama_33_70b_instr/wrapperbox

```

## WMT_EN_RU

```
python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/wmt_en_ru/llama_3.1_8B_Instr \
  --dataset_name Samsoup/Samsoup-WMT2020-ru-en \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/wmt_en_ru/llama_3.1_8B_Instr/wrapperbox
&& \
python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/wmt_en_ru/llama_33_70b_instr \
  --dataset_name Samsoup/Samsoup-WMT2020-ru-en \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/wmt_en_ru/llama_33_70b_instr/wrapperbox

```

## WMT_EN_ZH

```

python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/wmt_en_zh/llama_3.1_8B_Instr \
  --dataset_name Samsoup/Samsoup-WMT2020-en-zh \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/wmt_en_zh/llama_3.1_8B_Instr/wrapperbox
&& \
python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/wmt_en_zh/llama_33_70b_instr \
  --dataset_name Samsoup/Samsoup-WMT2020-en-zh \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/wmt_en_zh/llama_33_70b_instr/wrapperbox

```

## WMT_SI_EN

```

python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/wmt_si_en/llama_3.1_8B_Instr \
  --dataset_name Samsoup/Samsoup-WMT2020-si-en \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/wmt_si_en/llama_3.1_8B_Instr/wrapperbox
&& \
python run_wrapperbox_eval.py \
  --emb_dir /scratch/06782/ysu707/PRLE/wmt_si_en/llama_33_70b_instr \
  --dataset_name Samsoup/Samsoup-WMT2020-si-en \
  --output_dir /work/06782/ysu707/ls6/PRLE/results/wmt_si_en/llama_33_70b_instr/wrapperbox

```