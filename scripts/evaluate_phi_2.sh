#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 phi_2_zero_shot.py \
    --model_name "lvcalucioli/ca-finetuned-phi-2"

CUDA_VISIBLE_DEVICES=2 python3 phi_2_zero_shot.py \
    --model_name "microsoft/phi-2"