#!/bin/bash
    docker run --rm --gpus '"device='$CUDA_VISIBLE_DEVICES'"' -v $HOME:$HOME  sample_image_2 $HOME/sample_folder/tesi/finetuning_phi_2.sh