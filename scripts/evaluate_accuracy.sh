#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
#     --model_name "microsoft/phi-2" \
#     --predictions_path "./predictions/phi2_base_multiple_choice_0_8_5.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
#     --model_name "lvcalucioli/phi2_multiple-choice_merged" \
#     --predictions_path "./predictions/phi2_finetuned_multiple_choice_0_8_5.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
#     --model_name "lvcalucioli/phi2_question-answering_merged" \
#     --predictions_path "./predictions/phi2_finetuned_question_answering_on_multiple_choice_0_8_5.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
#     --model_name "microsoft/phi-2" \
#     --predictions_path "./predictions/phi2_base_multiple_choice_2_6_5.json" \


# CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
#     --model_name "lvcalucioli/phi2_okq_multiple-choice_merged" \
#     --predictions_path "./predictions/phi2_okq_multiple_choice_2_6_5.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
#     --model_name "lvcalucioli/phi2_linear_multiple-choice_merged" \
#     --predictions_path "./predictions/phi2_linear_multiple_choice_2_6_5.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
#     --model_name "lvcalucioli/phi2_linear_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_linear_syntetic_question-answering_on_multiple_choice_0_6_5.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
#     --model_name "lvcalucioli/flan-t5-large_multiple-choice" \
#     --predictions_path "./predictions/flan-t5-large_multiple-choice_0_6_5.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
#     --model_name "google/flan-t5-large" \
#     --predictions_path "./predictions/flan-t5-large_base_multiple-choice_0_6_5.json" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
#     --model_name "lvcalucioli/flan-t5-large__question-answering" \
#     --predictions_path "./predictions/flan-t5-large_finetuned_question-answering_on_multiple_choice_0_6_5.json" \

# CUDA_VISIBLE_DEVICES=2 python3 ../code/evaluate_accuracy.py \
#     --model_name "lvcalucioli/llamantino7b_2_question-answering_merged" \
#     --predictions_path "../predictions/llamantino_finetuned_question-answering_on_multiple_choice_0_6_5.json" \

# CUDA_VISIBLE_DEVICES=2 python3 ../code/evaluate_accuracy.py \
#     --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
#     --predictions_path "../predictions/llamantino_base_mutliple_choice.json" \
    
# CUDA_VISIBLE_DEVICES=2 python3 ../code/evaluate_accuracy.py \
#     --model_name "lvcalucioli/llamantino7b_2_multiple-choice_merged" \
#     --predictions_path "../predictions/llamantino_finetuned_mutliple_choice.json" \

CUDA_VISIBLE_DEVICES=2 python3 ../code/evaluate_accuracy.py \
    --model_name "lvcalucioli/zephyr-7b-beta_question-answering_question-answering_merged" \
    --predictions_path "../predictions/zephyr_finetuned_question-answering_on_mutliple_choice_0_6_5.json" \

CUDA_VISIBLE_DEVICES=2 python3 ../code/evaluate_accuracy.py \
    --model_name "lvcalucioli/zephyr-7b-beta_multiple_choice_merged" \
    --predictions_path "../predictions/zephyr_finetuned_mutliple_choice_on_multiple_choice_6_6_5.json" \

CUDA_VISIBLE_DEVICES=2 python3 ../code/evaluate_accuracy.py \
    --model_name "HuggingFaceH4/zephyr-7b-beta" \
    --predictions_path "../predictions/zephyr_base_mutliple_choice_0_6_5.json" \