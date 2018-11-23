import glob
import numpy as np
import os
import pickle
from pathlib import Path
import cv2
from tqdm import tqdm
from multiprocessing.dummy import Pool
from VideoUtils import CV2VideoCapture
from network.PoseEstimationUtils import plot_from_pose_coords, NUM_LIMBS
from PIL import Image


def getPoseEstimationImgFromCoordinatesByArr(oriImg, coords_arr):
    canvas, to_plot = plot_from_pose_coords(oriImg, coords_arr)
    return to_plot


mode = 'val'
img_output_dir = '/home/rabkinda/Documents/computer_vision/fencing/poses_clips_reduced_players_different_channel/img/' + mode
#pickle_output_dir = '/home/rabkinda/Documents/computer_vision/fencing/poses_clips_reduced_players_different_channel/pickle/' + mode
video_output_dir = '/home/rabkinda/Documents/computer_vision/fencing/poses_clips_reduced_players_different_channel/video/' + mode
clips_path = '/media/rabkinda/Gal_Backup/fencing/clips*/*.mp4'

clip_paths = [f for f in glob.glob(clips_path) if 'None' not in f]
seq_len = 60
img_shape = (256, 128)
pickle_file = Path('network/train_val_test_splitter/' + mode + '.pickle')
objects = []

if pickle_file.is_file():
    with open(pickle_file, 'rb') as f:
        objects = pickle.load(f)

max_parallel = 8
pool = Pool(max_parallel)
padding = 33


def generate_poses_clip(obj):
    curr_poses = obj[0]
    curr_poses = curr_poses.numpy()

    all_y = curr_poses[:, :, :, :, 1]
    all_y_non_zero = all_y[all_y > 0]
    if len(all_y_non_zero)>0:
        y_max = np.max(all_y_non_zero)
        y_min = np.min(all_y_non_zero)
    else:
        y_max = 0
        y_min = 0
    assert not np.isinf(y_max)
    assert not np.isinf(y_min)

    curr_clip_name = obj[2]
    # clip_path = [f for f in clip_paths if curr_clip_name in f][0]
    # cap = CV2VideoCapture(clip_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(video_output_dir, curr_clip_name) + '.mp4', fourcc, 20.0, img_shape)
    #res_array = np.zeros((seq_len, 2, 128, 256, 3))
    #pickle_file = Path(os.path.join(pickle_output_dir, curr_clip_name, 'data.pickle'))
    #if not os.path.exists(os.path.dirname(pickle_file)):
    #    os.makedirs(os.path.dirname(pickle_file))

    for seq_ind in range(seq_len):
        curr_fencing_players_coords = curr_poses[seq_ind]
        # curr_frame_img = cap.read()

        for p in range(len(curr_fencing_players_coords)):
            curr_frame_img = np.zeros((720 // 2, 1280 // 5, 3), dtype=np.uint8)
            curr_fencing_players_coords[p, :, :, 0] = curr_fencing_players_coords[p, :, :, 0] / 5.0
            curr_fencing_players_coords[p, :, :, 1] = curr_fencing_players_coords[p, :, :, 1] / 2.0

            curr_frame_with_chosen_pose = getPoseEstimationImgFromCoordinatesByArr(curr_frame_img,
                                                                                   np.expand_dims(
                                                                                       curr_fencing_players_coords[p],
                                                                                       axis=0))

            # pose_only_as_img = curr_frame_with_chosen_pose.astype(np.float32) - curr_frame_img.astype(np.float32)
            # pose_only_as_img = np.clip(pose_only_as_img, 0, 255)

            # poses_only_as_img_gray = cv2.cvtColor(poses_only_as_img, cv2.COLOR_RGB2GRAY)
            # ret, pose_only_as_img = cv2.threshold(pose_only_as_img, 0, 255, cv2.THRESH_BINARY)
            # poses_only_as_img_rgb = cv2.cvtColor(poses_only_as_img_gray, cv2.COLOR_GRAY2RGB)

            img = Image.fromarray(
                curr_frame_with_chosen_pose[max(0, int(y_min / 2) - padding):min(720 // 2,int(y_max / 2) + padding), :, :])
            img = img.resize(img_shape, Image.ANTIALIAS)  # ANTIALIAS

            #res_array[seq_ind, p] = np.array(img)

            curr_output_dir = os.path.join(img_output_dir, curr_clip_name, str(p))
            if not os.path.exists(curr_output_dir):
                os.makedirs(curr_output_dir)
            curr_output_path = os.path.join(curr_output_dir, str(seq_ind) + '.png')
            img.save(curr_output_path)
            out.write(np.array(img))

    #with open(pickle_file, 'wb') as f:
    #    pickle.dump(res_array, f)

    out.release()
    # cap.__del__()
    cv2.destroyAllWindows()


with tqdm(total=len(objects)) as pbar:
    for i, _ in enumerate(pool.imap_unordered(generate_poses_clip, objects)):
        pbar.update()
