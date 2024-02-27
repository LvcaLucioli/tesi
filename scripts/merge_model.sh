#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 ../code/merge_model.py \
    --base_model "HuggingFaceH4/zephyr-7b-beta" \
    --peft_model "lvcalucioli/zephyr-7b-beta_self-supervised" \

