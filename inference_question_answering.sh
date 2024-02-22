#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
    --model_name "HuggingFaceH4/zephyr-7b-beta" \
    --predictions_path "./predictions/zephyr_base_question-answering.json" \
    --start_prediction "<|assistant|>" \
    --end_prediction "<|system|>"

CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
    --model_name "lvcalucioli/zephyr-7b-beta_question-answering" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering.json" \
    --start_prediction "<|assistant|>" \
    --end_prediction "<|system|>"