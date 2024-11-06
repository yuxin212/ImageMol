#!/bin/bash

#SBATCH --account=pys1024
#SBATCH --job-name=imagemol_200m
#SBATCH --output=logs/imagemol_200m_%j.out
#SBATCH --error=logs/imagemol_200m_%j.err
#SBATCH --time=7-0
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=4
#SBATCH --mem=600gb
#SBATCH --partition=gpu
#SBATCH --mail-user=yangy9@ccf.org
#SBATCH --mail-type=ALL

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export EPOCHS=151
export NCCL_DEBUG=WARN

module purge

module load miniconda3
module load cuda/12.1.1

conda activate imagemol
cd /users/PYS1024/yangy9/ImageMol-yang

export UNIREF50_PATH=/users/PYS1024/yuxinyang/uniref50
export OUTPUT_PATH=/users/PYS1024/yuxinyang/trained_models/XL_model
mkdir -p $OUTPUT_PATH/ckpts
mkdir -p $OUTPUT_PATH/logs

srun python pretrain.py \
    --lr 0.01 \
    --wd -5 \
    --workers 96 \
    --epochs $EPOCHS \
    --batch 40960 \
    --dataset /fs/ess/PYS1024/data_pubchem_zinc_200m.csv \
    --ckpt_dir /fs/ess/PYS1024/imagemol_200m/ \
    --Jigsaw_lambda 1 \
    --cluster_lambda 0 \
    --contrastive_lambda 1 \
    --matcher_lambda 1 \
    --is_recover_training 1