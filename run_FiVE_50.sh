CUDA_VISIBLE_DEVICES=1 python preprocess_video_ddim.py \
    --data_dir data_FiVE \
    --latent_dir outputs_FiVE \
    --dataset_json data_FiVE/edit_prompt/edit1_FiVE.json \
    --max_number_of_frames 40 \
    --start_index 50
    
# CUDA_VISIBLE_DEVICES=3 python run_guidance.py \
#     --data_dir data_FiVE \
#     --latent_dir outputs_FiVE \
#     --dataset_json data_FiVE/edit_prompt/edit1_FiVE.json \
#     --max_number_of_frames 40

# CUDA_VISIBLE_DEVICES=3 python run_guidance.py \
#     --data_dir data_FiVE \
#     --latent_dir outputs_FiVE \
#     --dataset_json data_FiVE/edit_prompt/edit2_FiVE.json \
#     --max_number_of_frames 40

# CUDA_VISIBLE_DEVICES=3 python run_guidance.py \
#     --data_dir data_FiVE \
#     --latent_dir outputs_FiVE \
#     --dataset_json data_FiVE/edit_prompt/edit3_FiVE.json \
#     --max_number_of_frames 40

# CUDA_VISIBLE_DEVICES=3 python run_guidance.py \
#     --data_dir data_FiVE \
#     --latent_dir outputs_FiVE \
#     --dataset_json data_FiVE/edit_prompt/edit4_FiVE.json \
#     --max_number_of_frames 40