#!/bin/bash
#  salloc -q interactive -C gpu -t 4:00:00 -A nintern -N 1 --gpu-bind=none --image=nersc/pytorch:24.06.01 --module=gpu,nccl-plugin

#SBATCH -C gpu -N 1 -A your_account_name
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --image=nersc/pytorch:24.06.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH --output=out/%j.out
#SBATCH --time=28:00 -q debug

set -u  # exit if you try to use an uninitialized variable
set -e  # bash exits if any statement returns a non-true return value

# get inside
# shifter --image=nersc/pytorch:24.06.01 --module gpu,nccl-plugin --env PYTHONUSERBASE=$SCRATCH/cudaq
# pip3  install matplotlib==3.8.4 torch==2.0.1+cu118 torchvision==0.15.2+cu118 scikit-learn==1.4.2 -q --extra-index-url https://download.pytorch.org/whl/cu118
srun -l -N 1 shifter --env PYTHONUSERBASE=$SCRATCH/cudaq python hqml.py