#!/bin/bash -e
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --account=project_462000002

PROJECT_DIR=/scratch/project_462000008/tiks/Joonas
CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2.sif

c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

srun  \
  singularity exec \
    -B /var/spool/slurmd \
    -B /opt/cray \
    -B /usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjansson.so.4 \
    -B /usr/bin/hostname \
    -B $PROJECT_DIR \
    $CONTAINER bash run.sh
