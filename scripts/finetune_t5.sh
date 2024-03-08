#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python3 t5_finetuning.py \
#     --model_name "google/flan-t5-large" \
#     --new_model_name "flan-t5-large_" \
#     --num_train_epochs 5 \
#     --per_device_train_batch_size 8 \
#     --learning_rate 5e-5 \
#     --training_task "multiple-choice" \
#     --sample 1

# CUDA_VISIBLE_DEVICES=2 python3 ../code/t5_finetuning.py \
#     --model_name "google/flan-t5-large" \
#     --new_model_name "flan-t5-large" \
#     --num_train_epochs 8 \
#     --per_device_train_batch_size 1 \
#     --learning_rate 5e-5 \
#     --training_task "question-answering" \
#     --sample 1

CUDA_VISIBLE_DEVICES=2 python3 ../code/t5_finetuning.py \
    --model_name "google/flan-t5-large" \
    --new_model_name "flan-t5-large" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --training_task "multiple-choice" \
    --sample 1


# CUDA_VISIBLE_DEVICES=2 python3 t5_finetuning.py \
#     --model_name "google/flan-t5-large" \
#     --new_model_name "flan-t5-large_" \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 1 \
#     --learning_rate 3e-4 \
#     --training_task "multiple-choice" \
#     --sample 1

# CUDA_VISIBLE_DEVICES=2 python3 zephyr_finetuning.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --new_model_name "zephyr-7b-beta" \
#     --first_n 0 \
#     --num_train_epochs 12 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "multiple-choice" \
#     --sample 5

# CUDA_VISIBLE_DEVICES=2 python3 zephyr_finetuning.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_multiple-choice" \
#     --new_model_name "zephyr-7b-beta" \
#     --first_n 0 \
#     --num_train_epochs 12 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "multiple-choice" \
#     --sample 5
