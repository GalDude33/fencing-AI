import glob
import numpy as np
import os
import pickle
from pathlib import Path
import cv2
from tqdm import tqdm
from multiprocessing.dummy import Pool
from VideoUtils import CV2VideoCapture
from network.PoseEstimationUtils import plot_from_pose_coords


def getPoseEstimationImgFromCoordinatesByArr(oriImg, coords_arr):
    canvas, to_plot = plot_from_pose_coords(oriImg, coords_arr)
    return to_plot


mode = 'train'
output_dir = '/media/rabkinda/DATA/fencing/pose_estimation_checker/poses_clips/'+mode
clips_path = '/media/rabkinda/Gal_Backup/fencing/clips*/*.mp4'

clip_paths = [f for f in glob.glob(clips_path) if 'None' not in f]
seq_len = 60
img_shape = (1280, 720)
pickle_file = Path('network/train_val_test_splitter/' + mode + '.pickle')
objects = []

if pickle_file.is_file():
    with open(pickle_file, 'rb') as f:
        objects = pickle.load(f)

max_parallel = 8
pool = Pool(max_parallel)

def generate_poses_clip(obj):

    curr_poses = obj[0]
    curr_poses[:, :, :, :, 0] = curr_poses[:, :, :, :, 0] * 1280.0
    curr_poses[:, :, :, :, 1] = curr_poses[:, :, :, :, 1] * 720.0
    curr_poses = curr_poses.numpy()

    curr_clip_name = obj[2]
    clip_path = [f for f in clip_paths if curr_clip_name in f][0]
    cap = CV2VideoCapture(clip_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, curr_clip_name) + '.mp4', fourcc, 20.0, img_shape)

    for seq_ind in range(seq_len):
        curr_fencing_players_coords = curr_poses[seq_ind]
        curr_frame_img = cap.read()
        curr_frame_with_chosen_poses = getPoseEstimationImgFromCoordinatesByArr(curr_frame_img,
                                                                                curr_fencing_players_coords)
        poses_only_as_img = curr_frame_with_chosen_poses - curr_frame_img

        poses_only_as_img_gray = cv2.cvtColor(poses_only_as_img, cv2.COLOR_RGB2GRAY)
        ret, poses_only_as_img_gray = cv2.threshold(poses_only_as_img_gray, 0, 255, cv2.THRESH_BINARY)
        poses_only_as_img_rgb = cv2.cvtColor(poses_only_as_img_gray, cv2.COLOR_GRAY2RGB)

        out.write(poses_only_as_img_rgb)

    out.release()
    cap.__del__()
    cv2.destroyAllWindows()


with tqdm(total=len(objects)) as pbar:
    for i, _ in enumerate(pool.imap_unordered(generate_poses_clip, objects)):
        pbar.update()
