import pickle
import random
import os
from pathlib import Path
import cv2
import numpy as np
from VideoUtils import CV2VideoCapture
import glob
from network.PoseEstimationUtils import convert_points_to_lines, plot_from_pose_coords, IMG_SHAPE, \
    load_people_point_pose_arr


def getPoseEstimationImgFromCoordinatesByArr(oriImg, coords_arr, multiplyByImgSize=False):
    if multiplyByImgSize:
        coords_arr[:, :, :, 0] = coords_arr[:,:,:,0]*1280.0
        coords_arr[:, :, :, 1] = coords_arr[:, :, :, 1] * 720.0
    canvas, to_plot = plot_from_pose_coords(oriImg, coords_arr)
    #cv2.imwrite('result.png', to_plot)
    return to_plot


mode = 'val'
output_dir = '/media/rabkinda/DATA/fencing/pose_estimation_checker'
clips_path = '/media/rabkinda/Gal_Backup/fencing/clips*/*.mp4'
json_path = '/media/rabkinda/DATA/fencing/FinalPoseEstimationResults/jsons*/*.json'
pickle_file = Path('network/train_val_test_splitter/'+mode+'.pickle')
n = 5
img_shape = (1280, 720)

if pickle_file.is_file():
    with open(pickle_file, 'rb') as f:
        objects = pickle.load(f)

    chosen_objects = random.sample(objects, n)

    for i in range(n):
        chosen_object = chosen_objects[i]
        pose_point_coords_arr = chosen_object[0].numpy()

        clip_name = chosen_object[2]
        clip_path = [f for f in glob.glob(clips_path) if clip_name in f][0]
        json_paths = [f for f in glob.glob(json_path) if clip_name in f]
        json_paths.sort()

        cap = CV2VideoCapture(clip_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(output_dir, clip_name)+'.mp4', fourcc, 20.0, img_shape)
        out_all = cv2.VideoWriter(os.path.join(output_dir, clip_name+'_all')+'.mp4', fourcc, 20.0, img_shape)
        seq_len = pose_point_coords_arr.shape[0]

        for seq_ind in range(seq_len):
            curr_all_people_point_pose_arr, curr_all_people_point_pose_confidence_arr = load_people_point_pose_arr(json_paths[seq_ind])
            curr_all_people_point_pair_pose_arr = convert_points_to_lines(curr_all_people_point_pose_arr).astype(np.float32)

            curr_frame_img = cap.read()
            curr_frame_pose_coords = pose_point_coords_arr[seq_ind]
            curr_frame_with_chosen_poses = getPoseEstimationImgFromCoordinatesByArr(curr_frame_img, curr_frame_pose_coords, True)
            curr_frame_with_all_poses = getPoseEstimationImgFromCoordinatesByArr(curr_frame_img, curr_all_people_point_pair_pose_arr, False)
            out.write(curr_frame_with_chosen_poses)
            out_all.write(curr_frame_with_all_poses)

        out.release()
        out_all.release()
        cap.__del__()
        cv2.destroyAllWindows()
