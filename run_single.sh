CUDA_VISIBLE_DEVICES=3 python preprocess_video_ddim.py \
    --data_dir data \
    --latent_dir outputs \
    --dataset_json configs/dataset.json \
    --max_number_of_frames 40

CUDA_VISIBLE_DEVICES=3 python run_guidance.py \
    --data_dir data \
    --latent_dir outputs \
    --dataset_json configs/dataset.json \
    --max_number_of_frames 40