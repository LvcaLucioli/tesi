#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --predictions_path "./predictions/zephyr_base_question-answering.json" \
#     --start_prediction "<|assistant|>" \
#     --end_prediction "<|system|>" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 123

# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_question_answering.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_multiple-choice_merged" \
#     --predictions_path "../predictions/zephyr_finetuned_mutliple_choice_on_multiple_choice_8_8_15.json" \
#     --start_prediction "<|assistant|>" \
#     --end_prediction "<|system|>" \
#     --sample 5 \
#     --temperature 0.8 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 15

# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_question_answering.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --predictions_path "../predictions/zephyr_base_question-answering_8_8_15.json" \
#     --start_prediction "<|assistant|>" \
#     --end_prediction "<|system|>" \
#     --sample 5 \
#     --temperature 0.8 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 15


# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_question-answering_merged" \
#     --predictions_path "./predictions/zephyr_finetuned_question-answering_8_8_15.json" \
#     --start_prediction "<|assistant|>" \
#     --end_prediction "<|system|>" \
#     --sample 5 \
#     --temperature 0.8 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 15

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

CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_question_answering.py \
    --model_name "microsoft/phi-2" \
    --predictions_path "../predictions/phi2_base_question-answering_8_8_15.json" \
    --start_prediction "Output:" \
    --end_prediction "Instruct:" \
    --sample 5 \
    --temperature 0.8 \
    --top_p 0.8 \
    --top_k 20 \
    --num_beams 2 \
    --max_new_tokens 15

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
#     --predictions_path "./predictions/llamantino_base_question-answering_2_8_20.json" \
#     --start_prediction "[/INST]" \
#     --end_prediction "<</SYS>>" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 20

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
    # --model_name "lvcalucioli/llamantino7b_2_2_syntetic-question-answering_merged" \
    # --predictions_path "./predictions/llamantino_syntetic_question-answering_2_8_20.json" \
    # --start_prediction "[/INST]" \
    # --end_prediction "<</SYS>>" \
    # --sample 5 \
    # --temperature 0.2 \
    # --top_p 0.8 \
    # --top_k 20 \
    # --num_beams 2 \
    # --max_new_tokens 20

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/llamantino7b_2_2_syntetic-question-answering_3_merged" \
#     --predictions_path "./predictions/llamantino_syntetic_question-answering_3_2_8_20.json" \
#     --start_prediction "[/INST]" \
#     --end_prediction "<</SYS>>" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 20

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/flan-t5-large__question-answering" \
#     --predictions_path "./predictions/flan-t5-large_finetuned_question-answering_2_8_20.json" \
#     --start_prediction "Risposta:" \
#     --end_prediction "Domanda:" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 175

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "google/flan-t5-large" \
#     --predictions_path "./predictions/flan-t5-large_base_question-answering_2_8_20.json" \
#     --start_prediction "Risposta:" \
#     --end_prediction "Domanda:" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 175

# CUDA_VISIBLE_DEVICES=2 python3 ../code/inference_question_answering.py \
#     --model_name "lvcalucioli/phi2_question-answering_merged" \
#     --predictions_path "../predictions/phi2_finetuned_question-answering_8_8_15.json" \
#     --start_prediction "Output:" \
#     --end_prediction "Instruct:" \
#     --sample 5 \
#     --temperature 0.8 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 15