#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=01:00:00
#$ -N distributed_tutorial
#$ -o $JOB_ID.$JOB_NAME.log
#$ -j y


source /etc/profile.d/modules.sh
module load python/3.6/3.6.5 cuda/10.1/10.1.243 cudnn/7.6/7.6.5
source ~/venv/pytorch/bin/activate


MASTER_ADDR=`cat $PE_HOSTFILE | cut -f1 -d " " | head -1`
WORKERS=`cat $PE_HOSTFILE | cut -f1 -d " " | grep -v $MASTER_ADDR`
MASTER_PORT=7077
WORLD_SIZE=2
NPROCS_PER_NODE=4
NUM_NODES=2
RANK=0

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE
export RANK
export NPROCS_PER_NODE


python -m torch.distributed.launch \
   --nproc_per_node=4 \
	 --nnodes=2 \
	 --node_rank=0 \
	 --master_addr="$MASTER_ADDR" \
	 --master_port="$MASTER_PORT" \
   train_dist.py &


for worker in $WORKERS; do
  /system/uge/latest/bin/lx-amd64/qrsh -inherit $worker bash worker.sh —master_addr=$MASTER_ADDR —master_port=$MASTER_PORT &
done
wait