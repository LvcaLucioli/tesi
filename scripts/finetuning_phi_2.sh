#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python3 phi_supervised_finetuning.py \
#     --model_name "microsoft/phi-2" \
#     --new_model_name "phi2" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 4 \
#     --per_device_train_batch_size 4 \
#     --learning_rate 2e-4 \
#     --training_task "question-answering" \
#     --layers '["v_proj", "o_proj", "k_proj", "q_proj"]'

# CUDA_VISIBLE_DEVICES=2 python3 phi_supervised_finetuning.py \
#     --model_name "microsoft/phi-2" \
#     --new_model_name "phi2_linear_lm_head" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --learning_rate 2e-4 \
#     --training_task "multiple-choice"
    # --layers '["o_proj", "k_proj", "q_proj"]' 


# CUDA_VISIBLE_DEVICES=2 python3 phi_supervised_finetuning.py \
#     --model_name "microsoft/phi-2" \
#     --new_model_name "phi2" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --learning_rate 2e-5 \
#     --sample 1 \
#     --training_task "multiple-choice" \
#     --temperature 0.8 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2

CUDA_VISIBLE_DEVICES=2 python3 phi_supervised_finetuning.py \
    --model_name "microsoft/phi-2" \
    --new_model_name "phi2_linear" \
    --first_n 0 \
    --num_train_epochs 2 \
    --per_device_eval_batch_size 2 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-4 \
    --training_task "syntetic-question-answering" \
    # --layers '["v_proj", "k_proj", "q_proj", "q_proj"]'

# CUDA_VISIBLE_DEVICES=2 python3 merge_model.py \
#     --base_model "microsoft/phi-2" \
#     --peft_model "lvcalucioli/phi2_linear_syntetic-question-answering"

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/phi2_linear_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_linear_finetuned_syntetic_question-answering_2_8_15.json" \
#     --start_prediction "Output:" \
#     --end_prediction "Instruct:" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 15

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/phi2_linear_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_linear_finetuned_syntetic_question-answering_2_8_15.json" \

# CUDA_VISIBLE_DEVICES=2 python3 phi_supervised_finetuning.py \
#     --model_name "microsoft/phi-2" \
#     --new_model_name "phi2_vokq" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --learning_rate 2e-4 \
#     --training_task "syntetic-question-answering" \
#     --layers '["v_proj", "o_proj", "k_proj", "q_proj"]'

# CUDA_VISIBLE_DEVICES=2 python3 merge_model.py \
#     --base_model "microsoft/phi-2" \
#     --peft_model "lvcalucioli/phi2_vokq_syntetic-question-answering"

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/phi2_vokq_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_vokq_finetuned_syntetic_question-answering_2_8_15.json" \
#     --start_prediction "Output:" \
#     --end_prediction "Instruct:" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 15

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/phi2_vokq_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_vokq_finetuned_syntetic_question-answering_2_8_15.json" \



# CUDA_VISIBLE_DEVICES=2 python3 phi_supervised_finetuning.py \
#     --model_name "microsoft/phi-2" \
#     --new_model_name "phi2_vkq" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --learning_rate 2e-4 \
#     --training_task "syntetic-question-answering" \
#     --layers '["v_proj", "k_proj", "q_proj"]'

# CUDA_VISIBLE_DEVICES=2 python3 merge_model.py \
#     --base_model "microsoft/phi-2" \
#     --peft_model "lvcalucioli/phi2_vkq_syntetic-question-answering"

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/phi2_vkq_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_vkq_finetuned_syntetic_question-answering_2_8_15.json" \
#     --start_prediction "Output:" \
#     --end_prediction "Instruct:" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 15

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/phi2_vkq_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_vkq_finetuned_syntetic_question-answering_2_8_15.json" \



# CUDA_VISIBLE_DEVICES=2 python3 phi_supervised_finetuning.py \
#     --model_name "microsoft/phi-2" \
#     --new_model_name "phi2_voq" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --learning_rate 2e-4 \
#     --training_task "syntetic-question-answering" \
#     --layers '["v_proj", "o_proj", "q_proj"]'

# CUDA_VISIBLE_DEVICES=2 python3 merge_model.py \
#     --base_model "microsoft/phi-2" \
#     --peft_model "lvcalucioli/phi2_voq_syntetic-question-answering"

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/phi2_voq_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_voq_finetuned_syntetic_question-answering_2_8_15.json" \
#     --start_prediction "Output:" \
#     --end_prediction "Instruct:" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2 \
#     --max_new_tokens 15

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/phi2_voq_syntetic-question-answering_merged" \
#     --predictions_path "./predictions/phi2_voq_finetuned_syntetic_question-answering_2_8_15.json" \



# CUDA_VISIBLE_DEVICES=2 python3 zephyr_finetuning.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --new_model_name "zephyr-7b-beta" \
#     --first_n 0 \
#     --num_train_epochs 12 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "question-answering" \
#     --sample 5 \
#     --temperature 0.8 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2


# CUDA_VISIBLE_DEVICES=2 python3 zephyr_finetuning.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --new_model_name "zephyr-7b-beta" \
#     --first_n 0 \
#     --num_train_epochs 12 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "multiple-choice" \
#     --sample 5 \
#     --temperature 0.8 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2
