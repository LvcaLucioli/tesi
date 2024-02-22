#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 phi_supervised_finetuning.py \
    --model_name "microsoft/phi-2" \
    --new_model_name "phi2" \
    --first_n 0 \
    --num_train_epochs 12 \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-4 \
    --sample 1 \
    --training_task "question-answering" \
    --temperature 0.8 \
    --top_p 0.6 \
    --top_k 20 \
    --num_beams 2


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
