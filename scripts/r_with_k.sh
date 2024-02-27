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

# CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_merged" \
#     --predictions_path "./predictions/zephyr_finetuned_question-answering_8_8_15.json" \

# CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_merged" \
#     --predictions_path "./predictions/zephyr_finetuned_question-answering_8_8_10.json" \


# CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --predictions_path "./predictions/zephyr_base_question-answering.json" \

# CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
#     --model_name "microsoft/phi-2" \
#     --predictions_path "./predictions/phi2_base_question-answering_2_8_15.json" \

# CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
#     --model_name "lvcalucioli/phi2_question-answering_merged" \
#     --predictions_path "./predictions/phi2_finetuned_question-answering_2_8_15.json" \

# CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
#     --model_name "lvcalucioli/phi2_linear_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_linear_syntetic_question-answering_2_8_15.json" \
#     --encoder_name "BAAI/bge-large-en-v1.5"

# CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
#     --model_name "lvcalucioli/phi2_linear_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_linear_syntetic_question-answering_2_8_15.json" \
#     --encoder_name "dlicari/lsg16k-Italian-Legal-BERT"

CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
    --model_name "lvcalucioli/flan-t5-large__question-answering" \
    --predictions_path "./predictions/flan-t5-large_finetuned_question-answering_2_8_20.json" \
    --encoder_name "dlicari/lsg16k-Italian-Legal-BERT"

CUDA_VISIBLE_DEVICES=2 python3 r_with_k.py \
    --model_name "google/flan-t5-large" \
    --predictions_path "./predictions/flan-t5-large_base_question-answering_2_8_20.json" \
    --encoder_name "dlicari/lsg16k-Italian-Legal-BERT"


    --model_name "lvcalucioli/zephyr-7b-beta_multiple-choice_merged" \
    --predictions_path "../predictions/zephyr_finetuned_mutliple_choice_on_multiple_choice_8_8_123.json" \