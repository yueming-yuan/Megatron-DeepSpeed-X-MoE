#!/bin/bash

#SBATCH -A STF218
#SBATCH -J gpt1T_megatron
#SBATCH -o logs/gpt1T_megatron_128N_4x128-z1-fa2-%j.o
#SBATCH -e logs/gpt1T_megatron_128N_4x128-z1-fa2-%j.e
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 128

set +x
source /lustre/orion/world-shared/stf218/sajal/miniconda3-frontier/bin/activate
#conda activate /lustre/orion/stf006/world-shared/irl1/flash2
conda activate /lustre/orion/world-shared/stf218/sajal/flash2-copy
module load rocm/5.7.0
module load gcc/12.2.0

export ROCM_HOME=/opt/rocm-5.7.0

export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"
module load PrgEnv-gnu

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=INFO
# export settings
export TORCH_EXTENSIONS_DIR=$PWD/deepspeed
export HF_HOME=$PWD/hfdata
export OMP_NUM_THREADS=1

# setup hostfile
HOSTS=.hosts-job$SLURM_JOB_ID
HOSTFILE=hostfile.txt
srun hostname > $HOSTS
sed 's/$/ slots=8/' $HOSTS > $HOSTFILE

# setup env file
echo "PATH=$PATH" > .deepspeed_env
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> .deepspeed_env
echo "CPATH=$CPATH" >> .deepspeed_env
echo "TORCH_EXTENSIONS_DIR=$PWD/deepspeed" >> .deepspeed_env
echo "HF_HOME=$PWD/hfdata" >> .deepspeed_env
echo "ROCM_HOME=/opt/rocm-5.7.0" >> .deepspeed_env


scontrol show hostnames $SLURM_NODELIST > job.node.list
input="./job.node.list"
readarray -t arr <"$input"
first=${arr[0]}
echo "first=" $first
ips=`ssh $first hostname -I`
read -ra arr <<< ${ips}
export MASTER_ADDR=${arr[0]}
echo "MASTER_ADDR=" $MASTER_ADDR

ranks_per_node=8
gpus_per_rank=$((8/$ranks_per_node))
ranks_total=$(($ranks_per_node*$SLURM_JOB_NUM_NODES))

echo $ranks_per_node $gpus_per_rank $ranks_total
mkdir logs
mkdir logs/transformer

export CHECKPOINT_PATH=checkpoints/gpt2_345m
export VOCAB_FILE=gpt2-vocab.json
export MERGE_FILE=gpt2-merges.txt
export DATA_PATH=/lustre/orion/world-shared/stf218/sajal/mtds/gptdata/gpttext_article_document

export GPT_ARGS="--tensor-model-parallel-size 4 \
          --pipeline-model-parallel-size 128 \
          --num-layers 128 \
          --hidden-size 25600 \
          --num-attention-heads 200 \
          --seq-length 2048 \
          --max-position-embeddings 2048 \
  	  --micro-batch-size 1 \
	  --global-batch-size 1024 \
	  --train-samples 100000 \
       	  --lr-decay-samples 10000 \
          --lr-warmup-samples 100 \
          --lr 6.0e-5 \
	  --min-lr 6.0e-6 \
          --lr-decay-style cosine \
          --log-interval 1 \
          --eval-iters 1 \
          --eval-interval 1 \
	  --vocab-file $VOCAB_FILE \
	  --merge-file $MERGE_FILE \
          --split 98,2,0 \
          --clip-grad 1.0 \
	  --weight-decay 0.1 \
	  --adam-beta1 0.9 \
	  --adam-beta2 0.95 \
	  --init-method-std 0.006 \
          --bf16"



#OUTPUT_ARGS=<same as those in BERT pretraining above>
export OUTPUT_ARGS="--log-interval 1 \
             --eval-interval 1 \
             --eval-iters 1"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=2

#--use-flash-attn
#time srun -u -n$ranks_total -c 2 --gpus-per-task=$gpus_per_rank --gpu-bind=closest bash -c "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/sajal/software/aws-ofi-rccl/src/.libs/:${LD_LIBRARY_PATH}
#export NCCL_NET_GDR_LEVEL=3
#export FI_MR_CACHE_MONITOR=userfaultfd

time srun -u -n$ranks_total -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest bash -c "
source export_DDP_vars.sh 
python pretrain_gpt_deepspeed.py \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --data-path $DATA_PATH \
       --master-addr=$MASTER_ADDR \
       --data-path $DATA_PATH \
       --num-workers 1 \
       --deepspeed \
       --deepspeed_config ds_config_128N_1T_fp16.json \
       --deepspeed-activation-checkpointing \
       --checkpoint-activations \
       --zero-stage 1 \
       --use-flash-attn-v2
"
#--deepspeed-activation-checkpointing \
#       --checkpoint-activations \

       #--zero-stage 1 \
