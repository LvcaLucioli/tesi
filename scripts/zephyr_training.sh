#!/bin/bash
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
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2

# CUDA_VISIBLE_DEVICES=2 python3 zephyr_finetuning.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --new_model_name "zephyr-7b-beta" \
#     --first_n 10 \
#     --num_train_epochs 12 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "question-answering" \
#     --sample 5 \
#     --temperature 0.2 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2

# CUDA_VISIBLE_DEVICES=2 python3 zephyr_finetuning.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --new_model_name "zephyr-7b-beta" \
#     --first_n 10 \
#     --num_train_epochs 12 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "question-answering" \
#     --sample 5 \
#     --temperature 0.5 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2

# CUDA_VISIBLE_DEVICES=2 python3 zephyr_finetuning.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --new_model_name "zephyr-7b-beta" \
#     --first_n 10 \
#     --num_train_epochs 12 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "question-answering" \
#     --sample 5 \
#     --temperature 0.5 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2

# CUDA_VISIBLE_DEVICES=2 python3 zephyr_finetuning.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --new_model_name "zephyr-7b-beta" \
#     --first_n 100 \
#     --num_train_epochs 12 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "question-answering" \
#     --sample 5 \
#     --temperature 0.5 \
#     --top_p 0.2 \
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


CUDA_VISIBLE_DEVICES=2 python3 ../code/zephyr_finetuning.py \
    --model_name "HuggingFaceH4/zephyr-7b-beta" \
    --new_model_name "zephyr-7b-beta" \
    --first_n 0 \
    --num_train_epochs 5 \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-4 \
    --training_task "multiple-choice" \
    --sample 5 \
    --temperature 0.8 \
    --top_p 0.6 \
    --top_k 20 \
    --num_beams 2

# CUDA_VISIBLE_DEVICES=2 python3 ../code/zephyr_finetuning.py \
#     --model_name "HuggingFaceH4/zephyr-7b-beta" \
#     --new_model_name "zephyr-7b-beta" \
#     --first_n 0 \
#     --num_train_epochs 10 \
#     --per_device_eval_batch_size 4 \
#     --per_device_train_batch_size 4 \
#     --learning_rate 2e-4 \
#     --training_task "self-supervised" \

# CUDA_VISIBLE_DEVICES=2 python3 inference_question_answering.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering_merged" \
#     --predictions_path "./predictions/zephyr_finetuned_question-answering.json" \
#     --start_prediction "<|assistant|>" \
#     --end_prediction "<|system|>" \

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_rouge.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering" \
#     --predictions_path "./predictions/zephyr_finetuned_question-answering.json" \



# CUDA_VISIBLE_DEVICES=2 python3 zephyr_finetuning.py \
#     --model_name "lvcalucioli/zephyr-7b-beta_question-answering" \
#     --new_model_name "zephyr-7b-beta" \
#     --first_n 10 \
#     --num_train_epochs 12 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-4 \
#     --training_task "question-answering" \
#     --sample 5 \
#     --temperature 0.5 \
#     --top_p 0.8 \
#     --top_k 20 \
#     --num_beams 2