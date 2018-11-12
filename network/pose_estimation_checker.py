import random
import os
from pathlib import Path
import cv2
import numpy as np
from VideoUtils import CV2VideoCapture
import glob
from network.PoseEstimationUtils import convert_points_to_lines, plot_from_pose_coords, load_people_point_pose_arr, getFencingPlayersPoseArr


def getPoseEstimationImgFromCoordinatesByArr(oriImg, coords_arr, multiplyByImgSize=False):
    if multiplyByImgSize:
        coords_arr[:, :, :, 0] = coords_arr[:,:,:,0]*1280.0
        coords_arr[:, :, :, 1] = coords_arr[:, :, :, 1] * 720.0
    canvas, to_plot = plot_from_pose_coords(oriImg, coords_arr)
    #cv2.imwrite('result.png', to_plot)
    return to_plot

font = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText = (50, 110)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

mode = 'val'
output_dir = '/media/rabkinda/DATA/fencing/pose_estimation_checker'
clips_path = '/media/rabkinda/Gal_Backup/fencing/clips*/*.mp4'
json_path = '/media/rabkinda/DATA/fencing/FinalPoseEstimationResults/jsons*/*.json'
seq_len = 60
n = 5
img_shape = (1280, 720)
trg_clip_name = '00BoW1USRjA-41-T-21846'#None


if trg_clip_name is None:
    chosen_objects = random.sample(clips_path, n)
    chosen_objects = [Path(obj).stem for obj in chosen_objects]
else:
    chosen_objects = [trg_clip_name]

for i in range(len(chosen_objects)):
    clip_name = chosen_objects[i]

    clip_path = [f for f in glob.glob(clips_path) if clip_name in f][0]
    json_paths = [f for f in glob.glob(json_path) if clip_name in f]
    json_paths.sort()

    cap = CV2VideoCapture(clip_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, clip_name) + '.mp4', fourcc, 20.0, img_shape)
    out_all = cv2.VideoWriter(os.path.join(output_dir, clip_name + '_all') + '.mp4', fourcc, 20.0, img_shape)

    for seq_ind in range(seq_len):
        curr_fencing_players_coords = getFencingPlayersPoseArr(json_paths[seq_ind])

        curr_all_people_point_pose_arr, curr_all_people_point_pose_confidence_arr = load_people_point_pose_arr(json_paths[seq_ind])
        curr_all_people_point_pair_pose_arr = convert_points_to_lines(curr_all_people_point_pose_arr).astype(np.float32)

        curr_frame_img = cap.read()
        cv2.putText(curr_frame_img, str(seq_ind), topLeftCornerOfText, font, fontScale, fontColor, lineType)

        curr_frame_with_chosen_poses = getPoseEstimationImgFromCoordinatesByArr(curr_frame_img, curr_fencing_players_coords, True)
        curr_frame_with_all_poses = getPoseEstimationImgFromCoordinatesByArr(curr_frame_img, curr_all_people_point_pair_pose_arr, False)
        out.write(curr_frame_with_chosen_poses)
        out_all.write(curr_frame_with_all_poses)

    out.release()
    out_all.release()
    cap.__del__()
    cv2.destroyAllWindows()
