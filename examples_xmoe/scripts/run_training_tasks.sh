#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <framework> <num_gpus>"
    echo "Frameworks: x-moe, deepspeed-moe, tutel"
    exit 1
fi

framework=$1
num_gpus=$2

case $framework in
    x-moe)
        script="X-MoE-Small-node-1.sh"
        ;;
    deepspeed-moe)
        script="DeepSpeed-MoE-Small-node-1.sh"
        ;;
    tutel)
        script="Tutel-Small-node-1.sh"
        ;;
    *)
        echo "Invalid framework: $framework"
        echo "Supported: x-moe, deepspeed-moe, tutel"
        exit 1
        ;;
esac

batch_sizes=(1 2 4 8 16)

for bs in "${batch_sizes[@]}"; do
    ./$script $num_gpus $bs
done
