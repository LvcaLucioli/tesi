#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python3 llamantino_finetuning.py \
    # --model_name "lvcalucioli/llamantino7b_2_question-answering" \
    # --new_model_name "llamantino7b_2" \
    # --first_n 0 \
    # --num_train_epochs 14 \
    # --per_device_eval_batch_size 16 \
    # --per_device_train_batch_size 16 \
    # --learning_rate 2e-4 \
    # --training_task "question-answering" \
    # --sample 1 \
    # --temperature 0.5 \
    # --top_p 0.8 \
    # --top_k 20 \
    # --num_beams 2

CUDA_VISIBLE_DEVICES=2 python3 llamantino_finetuning.py \
    --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
    --new_model_name "llamantino7b_2_2" \
    --first_n 0 \
    --num_train_epochs 2 \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --training_task "syntetic-question-answering" \
    --sample 1 \
    --temperature 0.5 \
    --top_p 0.8 \
    --top_k 20 \
    --num_beams 2

CUDA_VISIBLE_DEVICES=2 python3 llamantino_finetuning.py \
    --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
    --new_model_name "llamantino7b_2_2" \
    --first_n 0 \
    --num_train_epochs 3 \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --training_task "syntetic-question-answering" \
    --sample 1 \
    --temperature 0.5 \
    --top_p 0.8 \
    --top_k 20 \
    --num_beams 2

# python3 llamantino_finetuning.py \
#     --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
#     --new_model_name "llamantino7b_2" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 4 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "multiple-choice" \
#     --sample 1 \
#     --temperature 0.5 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2

# CUDA_VISIBLE_DEVICES=2 python3 llamantino_finetuning.py \
#     --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
#     --new_model_name "llamantino7b_2" \
#     --first_n 0 \
#     --num_train_epochs 12 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "question-answering" \
#     --sample 1 \
#     --temperature 0.8 \
#     --top_p 0.6 \
#     --top_k 20 \
#     --num_beams 2


# CUDA_VISIBLE_DEVICES=1 python3 llamantino_finetuning.py \
#     --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
#     --new_model_name "llamantino7b_2" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 8 \
#     --per_device_train_batch_size 8 \
#     --learning_rate 2e-4 \
#     --training_task "question-answering" \
#     --sample 10

# CUDA_VISIBLE_DEVICES=1 python3 llamantino_finetuning.py \
#     --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
#     --new_model_name "llamantino7b_2_10" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "question-answering"

# CUDA_VISIBLE_DEVICES=1 python3 llamantino_finetuning.py \
#     --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
#     --new_model_name "llamantino7b_2_15" \
#     --first_n 0 \
#     --num_train_epochs 15 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "question-answering"

# CUDA_VISIBLE_DEVICES=1 python3 llamantino_finetuning.py \
#     --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
#     --new_model_name "llamantino7b_question_answering_finetuining_syntetic_1" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --learning_rate 2e-4 \

# CUDA_VISIBLE_DEVICES=1 python3 llamantino_finetuning.py \
#     --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
#     --new_model_name "llamantino7b_question_answering_finetuining_syntetic_3" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 3 \
#     --per_device_train_batch_size 3 \
#     --learning_rate 2e-4 \

# CUDA_VISIBLE_DEVICES=1 python3 llamantino_finetuning.py \
#     --model_name "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
#     --new_model_name "llamantino7b_question_answering_finetuining_syntetic_4" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 4 \
#     --per_device_train_batch_size 4 \
#     --learning_rate 2e-4 \

# CUDA_VISIBLE_DEVICES=1 python3 phi_supervised_finetuning.py \
#     --model_name "microsoft/phi-2" \
#     --new_model_name "phi2_question_answering_finetuining_syntetic_1" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 1 \
#     --per_device_train_batch_size 1 \
#     --learning_rate 2e-3 \

# CUDA_VISIBLE_DEVICES=1 python3 phi_supervised_finetuning.py \
#     --model_name "microsoft/phi-2" \
#     --new_model_name "phi2_question_answering_finetuining_syntetic_2" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 2 \
#     --per_device_train_batch_size 2 \
#     --learning_rate 2e-3 \

# CUDA_VISIBLE_DEVICES=1 python3 phi_supervised_finetuning.py \
#     --model_name "microsoft/phi-2" \
#     --new_model_name "phi2_question_answering_finetuining_syntetic_3" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 3 \
#     --per_device_train_batch_size 3 \
#     --learning_rate 2e-3 \

# CUDA_VISIBLE_DEVICES=1 python3 phi_supervised_finetuning.py \
#     --model_name "microsoft/phi-2" \
#     --new_model_name "phi2_question_answering_finetuining_syntetic_4" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 4 \
#     --per_device_train_batch_size 4 \
#     --learning_rate 2e-3 \
