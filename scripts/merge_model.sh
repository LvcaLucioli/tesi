#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 ../code/merge_model.py \
    --base_model "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
    --peft_model "lvcalucioli/llamantino7b_2_question-answering" \

