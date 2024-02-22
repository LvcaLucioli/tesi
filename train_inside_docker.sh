#!/bin/bash
    docker run -w faretra --rm --gpus '"device='$CUDA_VISIBLE_DEVICES'"' -v $HOME:$HOME  sample_image $HOME/sample_folder/train.sh