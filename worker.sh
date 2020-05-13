source /etc/profile.d/modules.sh
module load python/3.6/3.6.5 cuda/10.1/10.1.243 cudnn/7.6/7.6.5
source ~/venv/pytorch/bin/activate

NPROCS_PER_NODE=4
NUM_NODES=2
RANK=1
hostname

python -m torch.distributed.launch \
   --nproc_per_node=4 \
	 --nnodes=2 \
	 --node_rank=1 \
	 --master_addr="$MASTER_ADDR" \
	 --master_port="$MASTER_PORT" \
   train_dist.py &