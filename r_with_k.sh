#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
#     --model_name "phi2_question-answering"

# CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
#     --model_name "phi2_BASE_question-answering"

# CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
#     --model_name "llamantino7b_2_question-answering"

# CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
#     --model_name "llamantino7b_2_BASE_question-answering"

# CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
#     --model_name "phi2_BASE_question-answering"

CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
    --model_name "lvcalucioli/zephyr-7b-beta_question-answering_merged" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering.json" \

CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
    --model_name "HuggingFaceH4/zephyr-7b-beta" \
    --predictions_path "./predictions/zephyr_base_question-answering.json" \