#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
    --model_name "HuggingFaceH4/zephyr-7b-beta" \
    --predictions_path "./predictions/zephyr_base_question-answering.json" \

CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
    --model_name "lvcalucioli/zephyr-7b-beta_question-answering_merged" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering.json" \

