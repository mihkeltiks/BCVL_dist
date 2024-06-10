#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --output="output_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH --account=project_462000002

PROJECT_DIR=/scratch/project_462000008/tiks/BCVL_dist
CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2.sif



N=1 ; \  
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1) \

srun -N $N -n $((N*2)) --gpus $((N*2)) \
    singularity exec \
    -B $PWD \
    -B /usr/bin/hostname \
    -B /var/spool/slurmd \
    -B /opt/cray \
    -B /usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjansson.so.4 \
    /appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.2.sif  \
    ./run.sh
