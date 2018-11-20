import glob
import os
from multiprocessing.dummy import Pool
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from VideoUtils import CV2VideoCapture


mode = 'val'
poses_clips_path = '/media/rabkinda/DATA/fencing/pose_estimation_checker/poses_clips/'+mode
output_dir = '/home/rabkinda/Documents/computer_vision/fencing/poses_clips_reduced/'+mode

clip_names = [Path(f).stem for f in glob.glob(poses_clips_path+'/*') if 'None' not in f]
seq_len = 60
img_shape = (256, 128)

max_parallel = 16
pool = Pool(max_parallel)

def generate_reduced_poses_clip(clip_name):

    clip_path = os.path.join(poses_clips_path, clip_name + '.mp4')
    cap = CV2VideoCapture(clip_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, clip_name) + '.mp4', fourcc, 20.0, img_shape)

    for seq_ind in range(seq_len):
        curr_frame_img = cap.read()
        #curr_frame_img = cv2.cvtColor(curr_frame_img, cv2.COLOR_RGB2GRAY)
        img = Image.fromarray(curr_frame_img)
        img = img.resize(img_shape, Image.ANTIALIAS)
        out.write(np.array(img))

    out.release()
    cap.__del__()
    cv2.destroyAllWindows()


with tqdm(total=len(clip_names)) as pbar:
    for i, _ in enumerate(pool.imap_unordered(generate_reduced_poses_clip, clip_names)):
        pbar.update()
