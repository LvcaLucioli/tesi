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

CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
    --model_name "lvcalucioli/flan-t5-large_multiple-choice" \
    --predictions_path "./predictions/flan-t5-large_multiple-choice_0_6_5.json" \

CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
    --model_name "google/flan-t5-large" \
    --predictions_path "./predictions/flan-t5-large_base_multiple-choice_0_6_5.json" \

CUDA_VISIBLE_DEVICES=2 python3 evaluate_accuracy.py \
    --model_name "lvcalucioli/flan-t5-large__question-answering" \
    --predictions_path "./predictions/flan-t5-large_finetuned_question-answering_on_multiple_choice_0_6_5.json" \