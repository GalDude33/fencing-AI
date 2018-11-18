import glob
import os
import random
import re
from pathlib import Path

import torch
import cv2
import numpy as np
import torch.utils.data as torchdata
from VideoUtils import CV2VideoCapture
from network.utils import get_label_from_letter, flip_label
from PIL import Image


class Dataset(torchdata.Dataset):

    def __init__(self, mode, txt_path, poses_clips_path):
        self.seq_len = 60
        video_names_to_filter = [x.rstrip() for x in open(txt_path, 'r')]
        self.poses_clips_path = poses_clips_path
        vid_pose_files = [vid_pose_file for vid_pose_file in glob.glob(self.poses_clips_path + "/*.mp4")]
        self.objects = []

        for vid_pose_file in vid_pose_files:
            if any(vid_filter for vid_filter in video_names_to_filter if vid_filter in vid_pose_file):

                curr_clip_name, curr_clip_num, curr_label = self.getClipInfoFromFilename(vid_pose_file)

                if curr_label == 'T':
                    # if it is first clip of video, ignore it
                    #if int(curr_clip_num) == 0:
                    continue

                self.objects.append((curr_clip_name, curr_label))

        self.mode = mode

        if self.mode=='train':
            random.shuffle(self.objects)


    def __getitem__(self, index):

        clip_name, label = self.objects[index]
        clip_path = os.path.join(self.poses_clips_path, clip_name+'.mp4')
        cap = CV2VideoCapture(clip_path)
        frames = np.zeros((self.seq_len, int(720/2), int(1280/2)), dtype=np.float32)
        flip = random.choice([0, 1])

        for seq_ind in range(self.seq_len):
            curr_frame_img = cap.read()
            curr_frame_img = cv2.cvtColor(curr_frame_img, cv2.COLOR_RGB2GRAY)
            img = Image.fromarray(curr_frame_img)
            img = img.resize((int(1280/2), int(720/2)), Image.ANTIALIAS)
            curr_frame_img = np.array(img).astype(np.float32)
            curr_frame_img = curr_frame_img/255.0
            # [0, 1] => [-1, 1]
            curr_frame_img = (curr_frame_img*2)-1
            #maybe resize
            frames[seq_ind] = curr_frame_img


        if self.mode == 'train' and flip:
            label = flip_label(label)
            frames = frames[:, :, ::-1]

        return torch.from_numpy(frames.copy()), label, clip_name


    def __len__(self):
        return len(self.objects)


    def getClipInfoFromFilename(self, descriptor_file):
        curr_clip_name = Path(descriptor_file).stem
        curr_label = self.getLabelFromFilename(curr_clip_name)
        curr_clip_num = self.getClipNumberFromFilename(curr_clip_name)
        return curr_clip_name, curr_clip_num, curr_label


    def getLabelFromFilename(self, clip_name):
        result_letter = clip_name.split('-')[-2]
        return get_label_from_letter(result_letter)


    def getClipNumberFromFilename(self, clip_name):
        return clip_name.split('-')[-3]
