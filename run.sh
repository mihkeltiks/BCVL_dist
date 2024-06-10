#!/bin/bash

# Report affinity
echo "Rank \$SLURM_PROCID --> \$(taskset -p \$\$)"

# Report GPUs
if [ $SLURM_LOCALID -eq 0 ] ; then
    rocm-smi
else
  sleep 2
fi

# Start conda environment inside the container
$WITH_CONDA
# Setting the caches relevant to our application.
export TORCH_HOME=/workdir/torch-cache
export HF_HOME=/workdir/hf-cache
export TOKENIZERS_PARALLELISM=false

# Tell RCCL to use only Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

# Tell MIOpen where to store its cache
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-\$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

if [ $SLURM_LOCALID -eq 0 ] ; then
  rm -rf $MIOPEN_USER_DB_PATH
  mkdir -p $MIOPEN_USER_DB_PATH    
else
  sleep 2
fi

# export NCCL_DEBUG=INFO 
# export NCCL_DEBUG_SUBSYS=INIT,COLL
# export NCCL_DEBUG_FILE=/tmp/$(whoami)-rccl-rank\$SLURM_PROCID.txt

# Translate SLURM environment 

export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=1
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

set -x

# Run application
python -u train.py \
          --deepspeed \
          --deepspeed_config ds_fp16_z1_config.json \
          --batch-size $((1)) \
          --local_rank $SLURM_LOCALID \
          --world-size $SLURM_NPROCS \

