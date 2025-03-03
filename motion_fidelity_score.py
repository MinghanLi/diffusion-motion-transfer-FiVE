import argparse
from pathlib import Path

import os
import numpy as np
import torch
import imageio
from PIL import Image
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import read_video_from_path
from einops import rearrange
from omegaconf import OmegaConf


def get_similarity_matrix(tracklets1, tracklets2):
    displacements1 = tracklets1[:, 1:] - tracklets1[:, :-1]
    displacements1 = displacements1 / displacements1.norm(dim=-1, keepdim=True)

    displacements2 = tracklets2[:, 1:] - tracklets2[:, :-1]
    displacements2 = displacements2 / displacements2.norm(dim=-1, keepdim=True)

    similarity_matrix = torch.einsum("ntc, mtc -> nmt", displacements1, displacements2).mean(dim=-1)
    return similarity_matrix


def get_score(similarity_matrix):
    similarity_matrix_eye = similarity_matrix - torch.eye(similarity_matrix.shape[0]).to(similarity_matrix.device)
    # for each row find the most similar element
    max_similarity, _ = similarity_matrix_eye.max(dim=1)
    average_score = max_similarity.mean()
    return {
        "average_score": average_score.item(),
    }

def read_frames_from_dir(dir_path):
    """
    Read frames from a directory of images.

    Parameters:
    - dir_path (str): Path to the directory containing image frames.

    Returns:
    - np.ndarray: A NumPy array of frames, or None if the directory is empty or invalid.
    """
    try:
        # List all image files in the directory (sorted for consistent ordering)
        image_files = sorted(
            [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
        if not image_files:
            print(f"No image files found in directory: {dir_path}")
            return None

        # Load all images into a list
        frames = [imageio.imread(img) for img in image_files]
        return np.stack(frames)
    except Exception as e:
        print("Error reading frames from directory:", e)
        return None


def get_tracklets(model, video_path, mask=None, is_ori_video=False):
    if video_path.endswith('.mp4'):
        video = read_video_from_path(video_path)
    else:
        assert os.path.isdir(video_path), f'{video_path} must be a dir!'
        video = read_frames_from_dir(video_path) # t, h, w, 3
    
    video = video[:24]  # max frames 24
    
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().cuda()
    pred_tracks_small, pred_visibility_small = model(video, grid_size=55, segm_mask=mask)
    pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c ")
    return pred_tracks_small

def save_results_to_txt_file(file_path, content):
    """
    Append content to a text file, adding a new line for each entry.

    Parameters:
    - file_path (str): Path to the text file.
    - content (str): The content to save.
    """
    try:
        with open(file_path, "a") as file:  # Open in append mode
            file.write(content + "\n")  # Add the content followed by a newline
        print(f"Content saved to {file_path}.")
    except Exception as e:
        print(f"Error saving to file: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/motion_fidelity_score_config.yaml")
    parser.add_argument("--edit_video_path", type=str, default="results/locomotive_car/result_frames")
    parser.add_argument("--original_video_path", type=str, default="data/locomotive/")
    parser.add_argument("--output_path", type=str, default="results/locomotive_car")
    parser.add_argument("--output_path_txt", type=str, default="outputs/all_frames")
    parser.add_argument("--after_video_vae", type=bool, default=False)
    opt = parser.parse_args()
    config = OmegaConf.load(opt.config_path)
    config["edit_video_path"] = opt.edit_video_path
    config["original_video_path"] = opt.original_video_path
    config["output_path"] = opt.output_path

    model = CoTrackerPredictor(checkpoint=config.cotracker_model_path)
    model = model.cuda()

    video_path = config.edit_video_path
    original_video_path = config.original_video_path

    if config.use_mask:  # calculate trajectories only on the foreground of the video
        segm_mask = np.array(
            Image.open(config.mask_path)
        )
        segm_mask = torch.tensor(segm_mask).float() / 255
        # get bounding box mask from segmentation mask - rectangular mask that covers the segmentation mask
        box_mask = torch.zeros_like(segm_mask)
        minx = segm_mask.nonzero()[:, 0].min()
        maxx = segm_mask.nonzero()[:, 0].max()
        miny = segm_mask.nonzero()[:, 1].min()
        maxy = segm_mask.nonzero()[:, 1].max()
        box_mask[minx:maxx, miny:maxy] = 1
        box_mask = box_mask[None, None]
    else:
        box_mask = None

    edit_tracklets = get_tracklets(model, video_path, mask=box_mask)

    original_tracklets = get_tracklets(model, original_video_path, mask=box_mask)
    similarity_matrix = get_similarity_matrix(edit_tracklets, original_tracklets)
    similarity_scores_dict = get_score(similarity_matrix)
    
    save_dict = {
        "similarity_matrix": similarity_matrix.cpu(),
        "similarity_scores": similarity_scores_dict,
    }
    Path(config.output_path).mkdir(parents=True, exist_ok=True) 
    torch.save(save_dict, Path(config.output_path) / "metrics.pt")
    print("Motion similarity score: ", similarity_scores_dict["average_score"])

    save_results_to_txt_file(opt.output_path_txt, opt.edit_video_path + ' is: ' + str(similarity_scores_dict["average_score"]))