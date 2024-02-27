#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python3 zephyr_inference.py \
    --model_name "HuggingFaceH4/zephyr-7b-beta" \
    --hit_precision_log_path "zephyr_hit_precision.json" \
    --rouge_log_path "zephyr_rouge.json" \
    --accuracy_log_path "zephyr_accuracy.json" \
    --first_n 5 \
    --sample 3

CUDA_VISIBLE_DEVICES=1 python3 zephyr_inference.py \
    --model_name "lvcalucioli/zephyr_outputs" \
    --hit_precision_log_path "zephyr_hit_precision.json" \
    --rouge_log_path "zephyr_rouge.json" \
    --accuracy_log_path "zephyr_accuracy.json" \
    --first_n 5 \
    --sample 3