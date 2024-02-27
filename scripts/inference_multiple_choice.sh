#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python3 inference_multiple_choice.py \
#     --model_name "microsoft/phi-2" \
#     --predictions_path "./predictions/phi2_base_multiple_choice_2_6_5.json" \
#     --start_prediction "Output:" \
#     --end_prediction "Instruct:" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 5

# CUDA_VISIBLE_DEVICES=2 python3 inference_multiple_choice.py \
#     --model_name "lvcalucioli/phi2_okq_multiple-choice_merged" \
#     --predictions_path "./predictions/phi2_okq_multiple_choice_2_6_5.json" \
#     --start_prediction "Output:" \
#     --end_prediction "Instruct:" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 5
    
# CUDA_VISIBLE_DEVICES=2 python3 inference_multiple_choice.py \
#     --model_name "lvcalucioli/phi2_question-answering_merged" \
#     --predictions_path "./predictions/phi2_finetuned_question_answering_on_multiple_choice_0_8_5.json" \
#     --start_prediction "Output:" \
#     --end_prediction "Instruct:" \
#     --sample 5 \
#     --temperature 0.0 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 5

# CUDA_VISIBLE_DEVICES=2 python3 inference_multiple_choice.py \
#     --model_name "lvcalucioli/phi2_linear_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_linear_syntetic_question-answering_on_multiple_choice_2_6_5.json" \
#     --start_prediction "Output:" \
#     --end_prediction "Instruct:" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 5

# CUDA_VISIBLE_DEVICES=2 python3 inference_multiple_choice.py \
#     --model_name "lvcalucioli/phi2_linear_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_linear_syntetic_question-answering_on_multiple_choice_0_6_5.json" \
#     --start_prediction "Output:" \
#     --end_prediction "Instruct:" \
#     --sample 5 \
#     --temperature 0.0 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 5

# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_multiple_choice.py \
#     --model_name "lvcalucioli/llamantino7b_2_question-answering_merged" \
#     --predictions_path "../predictions/llamantino_finetuned_question-answering_on_multiple_choice_0_6_5.json" \
#     --start_prediction "### Response:" \
#     --end_prediction "### Instruction:" \
#     --sample 5 \
#     --temperature 0.0 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 5

# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_multiple_choice.py \
#     --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
#     --predictions_path "../predictions/llamantino_base_mutliple_choice.json" \
#     --start_prediction "### Response:" \
#     --end_prediction "### Instruction:" \
#     --sample 5 \
#     --temperature 0.0 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 5

# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_multiple_choice.py \
#     --model_name "lvcalucioli/llamantino7b_2_multiple-choice_merged" \
#     --predictions_path "../predictions/llamantino_finetuned_mutliple_choice.json" \
#     --start_prediction "### Response:" \
#     --end_prediction "### Instruction:" \
#     --sample 5 \
#     --temperature 0.0 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 5

CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_multiple_choice.py \
    --model_name "lvcalucioli/zephyr-7b-beta_question-answering_question-answering_merged" \
    --predictions_path "../predictions/zephyr_self_supervised_on_mutliple_choice_0_6_5.json" \
    --start_prediction "<|assistant|>" \
    --end_prediction "<|system|>" \
    --sample 5 \
    --temperature 0.0 \
    --top_p 0.6 \
    --top_k 20 \
    --num_beams 2 \
    --max_new_tokens 5

# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_multiple_choice.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --predictions_path "../predictions/zephyr_base_mutliple_choice_0_6_5.json" \
#     --start_prediction "<|assistant|>" \
#     --end_prediction "<|system|>" \
#     --sample 5 \
#     --temperature 0.0 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 5

# CUDA_VISIBLE_DEVICES=2 python3 inference_multiple_choice.py \
#     --model_name "google/flan-t5-large" \
#     --predictions_path "./predictions/flan-t5-large_base_multiple-choice_0_6_5.json" \
#     --start_prediction "Risposta:" \
#     --end_prediction "Domanda:" \
#     --sample 5 \
#     --temperature 0.0 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 5

# CUDA_VISIBLE_DEVICES=2 python3 inference_multiple_choice.py \
#     --model_name "lvcalucioli/phi2_linear_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_linear_syntetic_question-answering_on_multiple_choice_2_6_5.json" \
#     --start_prediction "Output:" \
#     --end_prediction "Instruct:" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 5