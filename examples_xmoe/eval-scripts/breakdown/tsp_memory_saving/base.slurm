#!/bin/bash

#SBATCH -A gen150
#SBATCH -J xmoe-eval
#SBATCH -o logs/%x-%j.o
#SBATCH -e logs/%x-%j.e
#SBATCH -t 0:10:00
#SBATCH -p batch
#SBATCH -N 128
#SBATCH -d singleton
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=yy28@illinois.edu

source /lustre/orion/gen150/scratch/pinaster/miniconda3/bin/activate
module load rocm/5.7.1.lua
module load libfabric/1.20.1
conda activate flash2
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.20.1/lib64:/lustre/orion/gen150/scratch/pinaster/software/rocm57/aws-ofi-rccl/src/.libs/:/opt/rocm-5.7.0/lib:$LD_LIBRARY_PATH
export TRITON_HIP_LLD_PATH=/opt/rocm-5.7.1/llvm/bin/ld.lld && export TRITON_CACHE_DIR="/lustre/orion/gen150/scratch/pinaster/triton_cache"


export ROCM_HOME=/opt/rocm-5.7.0

export LD_PRELOAD="/usr/lib64/libcrypto.so /usr/lib64/libssh.so.4 /usr/lib64/libssl.so.1.1"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# export NCCL_DEBUG=INFO
# export settings
export TORCH_EXTENSIONS_DIR=$PWD/deepspeed
export HF_HOME=$PWD/hfdata
export OMP_NUM_THREADS=1

export CHECKPOINT_PATH=checkpoints/smore-test
export VOCAB_FILE=/lustre/orion/gen150/scratch/pinaster/test2/Megatron-DeepSpeed-ORNL/gpt2-vocab.json
export MERGE_FILE=/lustre/orion/gen150/scratch/pinaster/test2/Megatron-DeepSpeed-ORNL/gpt2-merges.txt
export DATA_PATH=/lustre/orion/world-shared/stf218/sajal/mtds/gptdata/gpttext_article_document

NUM_GPUS=1024
BATCH_SIZE=1
# GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS * 32))
GLOBAL_BATCH_SIZE=1024
DIR=`pwd`
###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window
SEQ_LEN=4096

NUM_LAYERS=16 # TODO

## GPT-3 13B
MODEL_SIZE=6.7
HIDDEN_SIZE=7168
FFN_HIDDEN_SIZE=2048
NUM_ATTN_HEADS=128

# MODEL_SIZE=13
# NUM_LAYERS=2
# HIDDEN_SIZE=5120
# NUM_ATTN_HEADS=40


LR=1.2e-4
MIN_LR=1.0e-6

TRAIN_TOKENS=300000000000

TRAIN_ITERS=100

EXIT_DURATION=30000000
WARMUP_TOKENS=375000000
LR_DECAY_TOKENS=300000000000


MP_SIZE=1
PP_SIZE=1
EP_SIZE=256
EP_PARALLEL_SIZE=256

if [[ $EP_PARALLEL_SIZE -gt $NUM_GPUS ]]; then
    EP_PARALLEL_SIZE=$NUM_GPUS
fi

## Original GPT-3 model always set min LR at 10% of max LR. For MoE model, we
## found that lower LR and min LR (than the base dense model) helps.
## For 1.3B MoE-128 model we used LR=1.2e-4 and MIN_LR=1.0e-6.
## For 350M MoE-128 model we used LR=2.0e-4 and MIN_LR=2.0e-6, but they are not
## heavily tuned.
LR=1.2e-4
MIN_LR=1.0e-6

## Coefficient for MoE loss. We find that 0.01 is a good value at least for
## 1.3B MoE-128 model
MLC=0.01

## Below configs adjust the MoE expert token capacity limit during training and
## eval. To completely disable capacity limit, set MOE_DROP_TOKEN to false.
## Larger capacity factor or disabling capacity limit could improve training
## convergence, but will also reduce training throughput.
MOE_TRAIN_CAP_FACTOR=1.0
MOE_EVAL_CAP_FACTOR=1.0
MOE_MIN_CAP=4
MOE_DROP_TOKEN="true"
# MOE_DROP_TOKEN="false"
###############################################################################
### Curriculum learning (CL) configs
## Enable/disable CL
CL_ENABLED="false"
## Consult the tutorial https://www.deepspeed.ai/tutorials/curriculum-learning/
## for tuning the following configs
CL_START_SEQLEN=80
CL_AVG_SEQLEN=$(( (${CL_START_SEQLEN} + ${SEQ_LEN}) / 2 ))
CL_TOKENS=60
CL_TOKENS=$((${CL_TOKENS} * 1000000000))
CL_STEP=$(( ${CL_TOKENS} / (${GLOBAL_BATCH_SIZE} * ${CL_AVG_SEQLEN}) ))
###############################################################################
### Misc configs
LOG_INTERVAL=1
EVAL_ITERS=5
EVAL_INTERVAL=100
SAVE_INTERVAL=50

## Standard deviation for weight initialization
## We used 0.014 for 350M/1.3B dense/MoE models, and used 0.01 for 6.7B
## dense model. Usually larger model needs lower std.
INIT_STD=0.014
# INIT_STD=0.01

## Activation checkpointing saves GPU memory, but reduces training speed
# ACTIVATION_CHECKPOINT="true"
ACTIVATION_CHECKPOINT="false"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
NAME="gpt-${MODEL_SIZE}B-lr-${LR}-minlr-${MIN_LR}-bs-${GLOBAL_BATCH_SIZE}-gpus-${NUM_GPUS}-mp-${MP_SIZE}-pp-${PP_SIZE}-test"
if [[ $EP_SIZE -gt 1 ]]; then
    NAME="${NAME}-ep-${EP_SIZE}-mlc-${MLC}-cap-${MOE_TRAIN_CAP_FACTOR}-drop-${MOE_DROP_TOKEN}"
fi
if [ "${CL_ENABLED}" = "true" ]; then
    NAME="${NAME}-cl-${CL_START_SEQLEN}-${CL_STEP}"
fi

OUTPUT_BASEPATH=$DIR/output
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${host}_${current_time}"
mkdir -p ${TENSORBOARD_DIR} 
## Note that for MoE model with billion-scale base model, the checkpoint can be
## as large as TB-scale which normal NFS cannot handle efficiently.
#CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

# USE_INTERNAL_DATA="true"
USE_INTERNAL_DATA="false"

if [ "${USE_INTERNAL_DATA}" = "true" ]; then
    ## The internal data is only accessible within Microsoft
    ## For cluster Azure-EastUS-V100-32GB-4, Azure-WestUS3-A100
    # BASE_DATA_PATH=/vc_data/Megatron-LM/data
    # DATA_HOME="/vc_data/pile-cc1-cc2-shuf"
    ## For cluster Lab-RR1-V100
    BASE_DATA_PATH=/data/Megatron-LM/data
    DATA_HOME="/turing-ssd/users/conglli/data/pile-cc1-cc2-shuf"
    ## For cluster Azure-CentralUS-A100
    # BASE_DATA_PATH=/data/Megatron-LM/data
    # DATA_HOME=/vc_data_1/users/amawa/blended

    VOCAB_PATH=${BASE_DATA_PATH}/science-vocab.json
    MERGE_PATH=${BASE_DATA_PATH}/science-merges.txt
    ARX="${DATA_HOME}/ArXiv_ftfy_cleaned_id_shuf_text_document"
    BC2="${DATA_HOME}/BookCorpus2_ftfy_cleaned_id_shuf_text_document"
    B3="${DATA_HOME}/Books3_ftfy_cleaned_id_shuf_text_document"
    CC2020="${DATA_HOME}/CC-2020-50_id_cleaned_shuf_text_document"
    CC2021="${DATA_HOME}/CC-2021-04_id_cleaned_shuf_text_document"
    GIT="${DATA_HOME}/Github_ftfy_id_shuf_text_document"
    GUT="${DATA_HOME}/Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document"
    NIH="${DATA_HOME}/NIH_ExPorter_ftfy_id_shuf_text_document"
    OWT2="${DATA_HOME}/OpenWebText2_ftfy_cleaned_id_shuf_text_document"
    PCC="${DATA_HOME}/Pile-CC_id_cleaned_shuf_text_document"
    PM="${DATA_HOME}/PubMed_Abstracts_ftfy_id_shuf_text_document"
    RN="${DATA_HOME}/rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document"
    SE="${DATA_HOME}/StackExchange_ftfy_id_shuf_text_document"
    ST="${DATA_HOME}/stories_dedup0.7_shuf_cleaned_shuf_text_document"
    WIK="${DATA_HOME}/Wikipedia_en_ftfy_id_shuf_text_document"
    DATA_BLEND="0.14336 ${B3} 0.08962 ${RN} 0.19336 ${OWT2} 0.05689 ${SE} \
    0.00859 ${ST} 0.02897 ${PM} 0.04771 ${WIK} 0.00873 ${GUT} 0.01007 ${BC2} \
    0.00208 ${NIH} 0.13017 ${CC2020} 0.09446 ${PCC} 0.15652 ${CC2021} \
    0.01359 ${ARX} 0.01588 ${GIT}"
else
    VOCAB_PATH=/lustre/orion/gen150/scratch/pinaster/test2/Megatron-DeepSpeed-ORNL/gpt2-vocab.json
    MERGE_PATH=/lustre/orion/gen150/scratch/pinaster/test2/Megatron-DeepSpeed-ORNL/gpt2-merges.txt
    # Public the Pile dataset, can be downloaded at https://mystic.the-eye.eu/public/AI/pile_neox/
    DATA_BLEND=/data/the_pile_public_merged_nopreprocessing/pile_text_document
fi
###############################################################################
data_options=" \
         --vocab-file ${VOCAB_PATH} \
         --merge-file ${MERGE_PATH} \
         --data-path ${DATA_PATH}"
        # --data-impl mmap"
        
megatron_options=" \
        --override-opt_param-scheduler \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
        --num-experts ${EP_SIZE} \
        --moe-loss-coeff ${MLC} \
        --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
        --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
        --moe-min-capacity ${MOE_MIN_CAP} \
        --init-method-std ${INIT_STD} \
        --lr-decay-tokens ${LR_DECAY_TOKENS} \
        --lr-warmup-tokens ${WARMUP_TOKENS} \
        --micro-batch-size ${BATCH_SIZE} \
        --exit-duration-in-mins ${EXIT_DURATION} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-tokens ${TRAIN_TOKENS} \
        --train-iters ${TRAIN_ITERS} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 98,2,0 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 0 \
        --fp16 \
        --save ${CHECKPOINT_PATH} \
        --tensorboard-queue-size 1 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensorboard-dir ${TENSORBOARD_DIR} \
	--moe-train-capacity-factor 1.25 \
	--expert-interval 1 \
	--topk 8 \
        --use-uneven-all-to-all \
        --use-pft \
        --eval-iters 0 \
	--use-rotary-position-embeddings"
        #--enable-expert-sequence-parallelism \
	#--load ${CHECKPOINT_PATH}"
	#--use-rbd \

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
megatron_options="${megatron_options} \
        --checkpoint-activations"
fi

if [[ $EP_SIZE -gt 1 ]]; then
megatron_options="${megatron_options} \
        --create-moe-param-group"
fi

if [ "${MOE_DROP_TOKEN}" = "false" ]; then
megatron_options="${megatron_options} \
        --disable-moe-token-dropping"
fi

# megatron_options="${megatron_options} \
#         --checkpoint-intermediate"

WALL_CLOCK_BREAKDOWN="false"

# template_json="/lustre/orion/gen150/scratch/pinaster/smore/Megatron-DeepSpeed-ORNL/breakdown_scripts/ds_config_gpt_TEMPLATE_benchmark.json"
template_json="/lustre/orion/gen150/scratch/pinaster/moe-arch/system-benchmark/deepseek-style/ds_config_gpt_TEMPLATE.json"
#template_json="ds_config_gpt_Zero2_TEMPLATE.json"
config_json="ds_config_gpt_Zero0_${NAME}.json"
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/1/" \
    | sed "s/PRESCALE_GRAD/false/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
    | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
    | sed "s/WALL_CLOCK_BREAKDOWN/${WALL_CLOCK_BREAKDOWN}/" \
	  > ${config_json}

		    #--deepspeed_config ${config_json} \
deepspeed_options=" \
		    --deepspeed \
                    --deepspeed_config ${config_json} \
                    --no-pipeline-parallel"
		    #--pipeline-model-parallel-size ${PP_SIZE}"

# Currently MoE is not compatible with pipeline parallel
if [[ $EP_SIZE -gt 1 ]]; then
deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing \
        --checkpoint-activations"
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_NET_GDR_LEVEL=3
export FI_MR_CACHE_MONITOR=userfaultfd

readarray -t arr < <(scontrol show hostnames $SLURM_NODELIST)
first=${arr[0]}
echo "first=" $first

# Get the IP address of the first node
ips=$(ssh "$first" hostname -I)
read -ra arr <<< "$ips"
export MASTER_ADDR=${arr[0]}
echo "MASTER_ADDR=" $MASTER_ADDR

# megatron_options="${megatron_options} \
#         --checkpoint-attention \
#         --checkpoint-layernorm \
#         --checkpoint-intermediate"

# source export_DDP_vars.sh
srun -N 128 -n1024 -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest \
        python /lustre/orion/gen150/scratch/pinaster/moe-arch/system-benchmark/pretrain_gpt_deepspeed.py \
        ${megatron_options} \
        ${data_options} \
        ${deepspeed_options} \
        --master-addr=$MASTER_ADDR \
        --zero-stage 1 \
        --use-flash-attn \
        --use-flash-attn-v2 \
        2>&1 | tee n1024-base.log
echo ${run_cmd}
