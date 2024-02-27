#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_question-answering_merged" \
#     --predictions_path "./predictions/zephyr_finetuned_question-answering_8_8.json" \
#     --start_prediction "<|assistant|>" \
#     --end_prediction "<|system|>" \
#     --sample 5 \
#     --temperature 0.8 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 123

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_question-answering_merged" \
#     --predictions_path "./predictions/zephyr_finetuned_question-answering_5_8.json" \
#     --start_prediction "<|assistant|>" \
#     --end_prediction "<|system|>" \
#     --sample 5 \
#     --temperature 0.5 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 123

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_question-answering_merged" \
#     --predictions_path "./predictions/zephyr_finetuned_question-answering_2_2.json" \
#     --start_prediction "<|assistant|>" \
#     --end_prediction "<|system|>" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.2 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 123

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_question-answering_merged" \
#     --predictions_path "./predictions/zephyr_finetuned_question-answering_5_6.json" \
#     --start_prediction "<|assistant|>" \
#     --end_prediction "<|system|>" \
#     --sample 5 \
#     --temperature 0.5 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 123

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_question-answering_merged" \
#     --predictions_path "./predictions/zephyr_finetuned_question-answering_8_2.json" \
#     --start_prediction "<|assistant|>" \
#     --end_prediction "<|system|>" \
#     --sample 5 \
#     --temperature 0.8 \
#     --top_p 0.2 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 123







CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
    --model_name "lvcalucioli/phi2_question-answering_merged" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering_8_8.json" \

CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
    --model_name "lvcalucioli/phi2_question-answering_merged" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering_5_8.json" \

CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
    --model_name "lvcalucioli/phi2_question-answering_merged" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering_2_2.json" \

CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
    --model_name "lvcalucioli/phi2_question-answering_merged" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering_5_6.json" \

CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
    --model_name "lvcalucioli/phi2_question-answering_merged" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering_8_2.json" \






CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
    --model_name "lvcalucioli/phi2_question-answering_merged" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering_8_8.json" \

CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
    --model_name "lvcalucioli/phi2_question-answering_merged" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering_5_8.json" \

CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
    --model_name "lvcalucioli/phi2_question-answering_merged" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering_2_2.json" \

CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
    --model_name "lvcalucioli/phi2_question-answering_merged" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering_5_6.json" \

CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
    --model_name "lvcalucioli/phi2_question-answering_merged" \
    --predictions_path "./predictions/zephyr_finetuned_question-answering_8_2.json" \