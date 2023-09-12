#!/bin/bash
set -xe

if [ -e "result.txt" ]; then
    rm "result.txt"
fi

touch result.txt


for plugin in "fsdp" "low_level_zero" "torch_ddp_fp16"; do
   torchrun --standalone --nproc_per_node 1  benchmark.py --plugin $plugin --model_type "bert"
done
