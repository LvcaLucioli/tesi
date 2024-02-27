#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_with_context.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --predictions_path "zephyr_base.json" \
#     --max_new_tokens 32 \
#     --task "text-generation"

# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_with_context.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_question-answering_merged" \
#     --predictions_path "zephyr_finetuned_question_answering.json" \
#     --max_new_tokens 32 \
#     --task "text-generation" \
#     --start_predictions "## Risposta:"

# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_with_context.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_question-answering_merged" \
#     --predictions_path "zephyr_finetuned_question_answering_64.json" \
#     --max_new_tokens 64 \
#     --task "text-generation" \
#     --start_predictions "## Risposta:"





# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_with_context.py \
#     --model_name "lvcalucioli/phi2_linear_syntetic-question-answering_merged" \
#     --predictions_path "phi_syntetic_finetuned_question_answering_32.json" \
#     --max_new_tokens 32 \
#     --task "text-generation" \
#     --start_predictions "## Risposta:"

# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_with_context.py \
#     --model_name "lvcalucioli/phi2_question-answering_merged" \
#     --predictions_path "phi_finetuned_question_answering_32.json" \
#     --max_new_tokens 32 \
#     --task "text-generation" \
#     --start_predictions "## Risposta:"






# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_with_context.py \
#     --model_name "lvcalucioli/llamantino7b_2_question-answering_merged" \
#     --predictions_path "llamantino_finetuned_question_answering_32.json" \
#     --max_new_tokens 32 \
#     --task "text-generation" \
#     --start_predictions "## Risposta:"
    
# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_with_context.py \
#     --model_name "lvcalucioli/llamantino7b_2_2_syntetic-question-answering_merged" \
#     --predictions_path "llamantino_syntetic_finetuned_question_answering_32.json" \
#     --max_new_tokens 32 \
#     --task "text-generation" \
#     --start_predictions "## Risposta:"


CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_with_context.py \
    --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
    --predictions_path "llamantino_base.json" \
    --max_new_tokens 32 \
    --task "text-generation" \
    --start_predictions "## Risposta:"