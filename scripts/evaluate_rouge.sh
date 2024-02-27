#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --predictions_path "./predictions/zephyr_base_question-answering.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_merged" \
#     --predictions_path "./predictions/zephyr_finetuned_question-answering_8_8_15.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_merged" \
#     --predictions_path "./predictions/zephyr_finetuned_question-answering_8_8_10.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "microsoft/phi-2" \
#     --predictions_path "./predictions/phi2_base_question-answering_2_8_15.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/phi2_linear_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_linear_syntetic_question-answering_2_8_15.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
#     --predictions_path "./predictions/llamantino_base_question-answering_2_8_20.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/llamantino7b_2_question-answering_merged" \
#     --predictions_path "./predictions/llamantino_finetuned_question-answering_2_8_20.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/flan-t5-large__question-answering" \
#     --predictions_path "./predictions/flan-t5-large_finetuned_question-answering_2_8_20.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "google/flan-t5-large" \
#     --predictions_path "./predictions/flan-t5-large_base_question-answering_2_8_20.json" \


# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/llamantino7b_2_2_syntetic-question-answering_3_merged" \
#     --predictions_path "./predictions/llamantino_syntetic_question-answering_3_2_8_20.json" \

# CUDA_VISIBLE_DEVICES=2 python3 ../code/evaluate_rouge.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_multiple-choice_merged" \
#     --predictions_path "../predictions/zephyr_finetuned_mutliple_choice_on_multiple_choice_8_8_123.json" \

# CUDA_VISIBLE_DEVICES=2 python3 ../code/evaluate_rouge.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_multiple_choice_merged" \
#     --predictions_path "../predictions/zephyr_finetuned_mutliple_choice_on_multiple_choice_8_8_15.json" \

# CUDA_VISIBLE_DEVICES=2 python3 ../code/evaluate_rouge.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --predictions_path "../predictions/zephyr_base_question-answering_8_8_15.json" \

CUDA_VISIBLE_DEVICES=2 python3 ../code/evaluate_rouge.py \
    --model_name "lvcalucioli/zephyr-7b-beta_question-answering_question-answering_merged" \
    --predictions_path "../predictions/inference_with_context/zephyr_finetuned_question_answering.json" \
    
# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/phi2_question-answering_merged" \
#     --predictions_path "./predictions/phi2_finetuned_question-answering_8_8.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/phi2_question-answering_merged" \
#     --predictions_path "./predictions/phi2_finetuned_question-answering_2_2.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/phi2_question-answering_merged" \
#     --predictions_path "./predictions/phi2_finetuned_question-answering_5_6.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/phi2_question-answering_merged" \
#     --predictions_path "./predictions/phi2_finetuned_question-answering_8_2.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "microsoft/phi-2" \
#     --predictions_path "./predictions/phi2_base_question-answering.json" \