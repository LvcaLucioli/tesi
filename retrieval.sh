#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 retrieval.py \
    --encoder_model_name "dlicari/Italian-Legal-BERT" \
    --generator_model_name "google/flan-t5-large" \
    --num_chunks_to_retrieve 3 \