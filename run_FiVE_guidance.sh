#!/bin/bash

while true; do

    CUDA_VISIBLE_DEVICES=0 python run_guidance.py \
        --data_dir data_FiVE \
        --latent_dir outputs_FiVE \
        --dataset_json data_FiVE/edit_prompt/edit1_FiVE.json \
        --max_number_of_frames 40

    CUDA_VISIBLE_DEVICES=0 python run_guidance.py \
        --data_dir data_FiVE \
        --latent_dir outputs_FiVE \
        --dataset_json data_FiVE/edit_prompt/edit2_FiVE.json \
        --max_number_of_frames 40

    # CUDA_VISIBLE_DEVICES=0 python run_guidance.py \
    #     --data_dir data_FiVE \
    #     --latent_dir outputs_FiVE \
    #     --dataset_json data_FiVE/edit_prompt/edit3_FiVE.json \
    #     --max_number_of_frames 40

    # CUDA_VISIBLE_DEVICES=0 python run_guidance.py \
    #     --data_dir data_FiVE \
    #     --latent_dir outputs_FiVE \
    #     --dataset_json data_FiVE/edit_prompt/edit4_FiVE.json \
    #     --max_number_of_frames 40

done