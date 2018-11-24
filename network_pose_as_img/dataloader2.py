import glob
import os
import random
from pathlib import Path
import torch
import cv2
import numpy as np
import torch.utils.data as torchdata
import torchvision
from random import uniform
from VideoUtils import CV2VideoCapture
from network.utils import get_label_from_letter, flip_label
from PIL import Image
from pyflow import pyflow
from scipy.ndimage import zoom


class Dataset(torchdata.Dataset):

    def __init__(self, mode, txt_path, poses_path, filtered_seq_len, filtered_seq_step_size, use_optical_flow, players_in_same_channel):
        self.seq_len = 60
        self.filtered_seq_len = filtered_seq_len
        self.filtered_seq_step_size = filtered_seq_step_size
        self.use_optical_flow = use_optical_flow
        self.players_in_same_channel = players_in_same_channel

        video_names_to_filter = [x.rstrip() for x in open(txt_path, 'r')]
        self.poses_clips_path = poses_path
        vid_pose_files = [vid_pose_file for vid_pose_file in glob.glob(self.poses_clips_path + "/*.mp4")]
        self.objects = []

        for vid_pose_file in vid_pose_files:
            if any(vid_filter for vid_filter in video_names_to_filter if vid_filter in vid_pose_file):

                curr_clip_name, curr_clip_num, curr_label = self.getClipInfoFromFilename(vid_pose_file)

                #if curr_label == 2:
                #    # if it is first clip of video, ignore it
                #    #if int(curr_clip_num) == 0:
                #    continue

                self.objects.append((curr_clip_name, curr_label))

        self.mode = mode

        if self.mode=='train':
            random.shuffle(self.objects)


    def __getitem__(self, index):

        clip_name, label = self.objects[index]
        clip_path = os.path.join(self.poses_clips_path, clip_name + '.mp4')
        cap = CV2VideoCapture(clip_path)
        trg_people_channel_num = 1 if self.players_in_same_channel else 2
        frames = np.zeros((self.filtered_seq_len, trg_people_channel_num, 128, 256, 3), dtype=np.uint8)
        flow = np.zeros((self.filtered_seq_len, trg_people_channel_num, 128, 256, 2), dtype=np.float32)
        angle, translate, scale = 0.0, 0.0, 1.0

        if self.mode == 'train':
            flip = random.choice([0, 1])
            angle, translate, scale = self.get_augmentation_params(angle_max=15, translate_max=((-10, 10), (-20, 10)), scale_range=(0.75, 1.15))

        seqs_to_count = [i for i in range(self.seq_len) if (i >= 0 and i <= 52 and i % self.filtered_seq_step_size == 0)]
        seqs_to_count = seqs_to_count[-self.filtered_seq_len:]  # filter sequence
        seqs_to_count.sort()

        for seq_ind in range(self.seq_len):
            for p in [0,1]:
                curr_frame_img = cap.read()

                if seq_ind in seqs_to_count:
                    if self.mode == 'train':
                        img = Image.fromarray(curr_frame_img)
                        # augmentations
                        img = torchvision.transforms.functional.affine(img, angle=angle, translate=translate, scale=scale, shear=0, resample=0, fillcolor=0)
                        curr_frame_img = np.array(img)
                    frames[seqs_to_count.index(seq_ind), p % trg_people_channel_num] += curr_frame_img

        if self.mode == 'train' and flip:
            label = flip_label(label)
            frames = frames[:, :, :, ::-1, :]

        if self.use_optical_flow:
            for i in range(self.filtered_seq_len - 1):
                for p in range(trg_people_channel_num):
                    flow[i, p] = self.calculate_optical_flow(frames[i, p], frames[i + 1, p])

        frames = frames.astype(np.float32)
        frames = frames/255.0
        # [0, 1] => [-1, 1]
        frames = (frames*2)-1

        return torch.from_numpy(frames.copy()).unsqueeze(2).transpose(2,-1).squeeze(-1), \
               torch.from_numpy(flow).unsqueeze(2).transpose(2,-1).squeeze(-1),\
               label, clip_name


    def __len__(self):
        return len(self.objects)


    def calculate_optical_flow(self, prev_frame, next_frame):
        #prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        #next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        alpha = 0.012
        ratio = 1 #0.75
        minWidth = int(180/4)
        nOuterFPIterations = 1
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
        img_shape = (int(256/4), int(128/4))

        prev_img = Image.fromarray(prev_frame)
        prev_img = prev_img.resize(img_shape, Image.ANTIALIAS)

        next_img = Image.fromarray(next_frame)
        next_img = next_img.resize(img_shape, Image.ANTIALIAS)

        u, v, im2W = pyflow.coarse2fine_flow(
            np.array(prev_img).astype(float)/255.0, np.array(next_img).astype(float)/255.0, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)

        flow = np.concatenate((u[..., None], v[..., None]), axis=2)

        #flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #flow[prev_frame_gray < 15] = 0
        flow = flow * 4.0

        flow = zoom(flow, (4, 4, 1))
        return flow.astype(np.float32)


    def get_augmentation_params(self, angle_max, translate_max, scale_range):
        scale = uniform(0, scale_range[1]-scale_range[0]) + scale_range[0]  # 0.75-1.15
        angle = uniform(-angle_max, angle_max)
        translate_max_x = translate_max[0]
        translate_max_y = translate_max[1]
        translate = (uniform(translate_max_x[0], translate_max_x[1]), uniform(translate_max_y[0], translate_max_y[1]))
        return angle, translate, scale


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
