#!/bin/bash
#SBATCH --comment clap
#SBATCH --partition=gamma
#SBATCH --job-name=mclap
#SBATCH --nodes 3
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=6
#SBATCH --exclusive
#SBATCH --output=%x_%j.out

#module load cuda/11.3.1
#module load mpi
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export NCCL_DEBUG=info
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_GID_INDEX=3
export WORLD_SIZE=8
# export MASTER_PORT=12802



OMP_NUM_THREADS=20 torchrun --nnodes=1 --nproc_per_node 4 --master_addr=localhost --master_port=2120 -m training.main \
    --name "testing" \
    --save-frequency 2 \
    --save-top-performance 3 \
    --save-most-recent \
    --dataset-type="csv" \
    --precision="fp32" \
    --batch-size=64 \
    --lr=5e-4 \
    --wd=0.0 \
    --epochs=30 \
    --workers=6 \
    --use-bn-sync \
    --amodel HTSAT-base \
    --tmodel t5 \
    --warmup 3200 \
    --report-to "wandb" \
    --wandb-notes "test" \
    --train-data /fs/nexus-projects/brain_project/aaai_2025/icassp_2025/icassp_rewrite/rewritten.csv \
    --val-data /fs/nexus-projects/brain_project/naacl_audio/val.csv  \
    --top-k-checkpoint-select-dataset="Clotho-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --openai-model-cache-dir ./cache \
    --logs /fs/gamma-projects/audio/clap_logs \
    --seed 3407 \
    --gather-with-grad \
    --optimizer "adam" \
    --data-filling "repeatpad" \
    --data-truncating "fusion" \
    --enable-fusion \
    --fusion-type "aff_2d" \
    --pretrained-audio /fs/nexus-projects/brain_project/CLAP/htsat_base_zaratan.ckpt
