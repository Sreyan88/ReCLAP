#!/bin/bash
# set -x

export HF_TOKEN='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'
export HF_HOME=/fs/nexus-projects/brain_project/microsoft/cache

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

input_train_file='/fs/nexus-projects/brain_project/naacl_audio/data_on_nexus_vecaps_fma_fsd.csv'

python split_csv.py --file_path $input_train_file --n $gpu_count --output_path /fs/nexus-projects/brain_project/aaai_2025/icassp_2025/icassp_rewrite/

input_train_file='/fs/nexus-projects/brain_project/aaai_2025/icassp_2025/icassp_rewrite/data_on_nexus_vecaps_fma_fsd_2.csv'

files=()
for i in $(seq 0 $(($gpu_count - 1))); do
    files+=("${input_train_file%.csv}_$i.csv")
done

gpus=()
for i in $(seq 0 $(($gpu_count - 1))); do
    gpus+=($i)
done

for i in $(seq 0 $(($gpu_count-1))); do
    CUDA_VISIBLE_DEVICES=${gpus[$i]} python llama_rephrase_2.py --file_path ${files[$i]} --iteration "$i" &
done



