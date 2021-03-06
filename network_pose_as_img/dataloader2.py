import glob
import gzip
import os
import pickle
import random
from multiprocessing.dummy import Pool
from pathlib import Path
import torch
import cv2
import numpy as np
import torch.utils.data as torchdata
import torchvision
from random import uniform

from torchvision.transforms.functional import _get_inverse_affine_matrix

from VideoUtils import CV2VideoCapture
from network.PoseEstimationUtils import getFencingPlayersPoseArr
from network.utils import get_label_from_letter, flip_label
from PIL import Image
from pyflow import pyflow
from scipy.ndimage import zoom, affine_transform

from optical_flow_from_pose import pose2flow, draw_hsv

padding = 33


class Dataset(torchdata.Dataset):

    def __init__(self, mode, txt_path, poses_path, filtered_seq_len, filtered_seq_step_size, use_optical_flow,
                 use_pose_optical_flow, players_in_same_channel, pose_jsons_dir=None):
        self.use_pose_optical_flow = use_pose_optical_flow
        self.seq_len = 60
        self.filtered_seq_len = filtered_seq_len
        self.filtered_seq_step_size = filtered_seq_step_size
        self.use_optical_flow = use_optical_flow
        self.players_in_same_channel = players_in_same_channel

        video_names_to_filter = [x.rstrip() for x in open(txt_path, 'r')]
        self.poses_clips_path = poses_path
        self.clip_pose_file_paths = set([vid_pose_file for vid_pose_file in glob.glob(self.poses_clips_path + "/*/*.mp4")])
        self.objects = []

        self.clip_name2path_dict={}
        for curr_clip_path in self.clip_pose_file_paths:
            curr_clip_name, curr_clip_num, curr_label = self.getClipInfoFromFilename(curr_clip_path)
            self.clip_name2path_dict[curr_clip_name] = (curr_clip_path, curr_label)

        for curr_clip_name in video_names_to_filter:
            curr_label = self.clip_name2path_dict[curr_clip_name][1]
            self.objects.append((curr_clip_name, curr_label))

        #self.poses_dict = {}
        if pose_jsons_dir is not None:
            pickle_file = Path('clip2pose.pkl.gz')
            if pickle_file.is_file():
                with gzip.open(pickle_file, 'rb') as f:
                    self.poses_dict = pickle.load(f)
            else:
                descriptor_files = list(glob.glob(pose_jsons_dir + "/*/*.json"))
                clip_names = set([os.path.join(Path(_).parent.stem,Path(_).stem[:-len('-00_keypoints')]) for _ in descriptor_files])

                padding = 33
                #def fix_poses(poses):
                #    all_y = poses[:, :, :, :, 1]
                #    all_y_non_zero = all_y[all_y > 0]
                #    if len(all_y_non_zero) > 0:
                #        y_max = np.max(all_y_non_zero)
                #        y_min = np.min(all_y_non_zero)
                #    else:
                #        y_max = 0
                #        y_min = 0
                #    poses[:, :, :, 0] = poses[:, :, :, 0] / 5.0
                #    poses[:, :, :, 1] = poses[:, :, :, 1] / 2.0

                #    poses[:, :, :, 1] = poses[:, :, :, 1] - max(0, int(y_min / 2) - padding)

                #    poses[:, :, :, 1] = poses[:, :, :, 1] * 128 // (min(720 // 2,int((y_max / 2) + padding))

                #    return poses

                def clip_name_to_pose(clip_name):
                    curr_descriptors = [os.path.join(pose_jsons_dir, '{}-{:02d}_keypoints.json').format(clip_name, i) for i in range(1, 60 + 1)]
                    poses = getFencingPlayersPoseArr(curr_descriptors)
                    #poses = fix_poses(poses)
                    return (clip_name.split('/')[-1], poses)
                with Pool() as p:
                    self.poses_dict = dict(p.map(clip_name_to_pose, clip_names))
                with gzip.open(pickle_file, 'wb') as f:
                    pickle.dump(self.poses_dict, f)

        self.mode = mode

        if self.mode == 'train':
            random.shuffle(self.objects)


    def __getitem__(self, index):
        clip_name, label = self.objects[index]
        clip_path = self.clip_name2path_dict[clip_name][0]
        cap = CV2VideoCapture(clip_path)
        trg_people_channel_num = 1 if self.players_in_same_channel else 2
        frames = np.zeros((self.filtered_seq_len, trg_people_channel_num, 128, 256, 3), dtype=np.uint8)
        flow = np.zeros((self.filtered_seq_len, trg_people_channel_num, 128, 256, 2), dtype=np.float32)
        angle, translate, scale, shear = 0.0, 0.0, 1.0, 0.0

        if self.mode == 'train':
            flip = random.choice([0, 1])
            angle, translate, scale, shear = self.get_augmentation_params(angle_max=15, translate_max=((-10, 10), (-20, 10)),
                                                                   scale_range=(0.75, 1.15), shear_max=10)
            center = (128 * 0.5 + 0.5, 256 * 0.5 + 0.5)
            affine_matrix = np.eye(3)
            affine_matrix[:, :2] = np.array(_get_inverse_affine_matrix(center, angle, translate, scale, shear=shear)).reshape(3,2)
            #print(affine_matrix)

        seqs_to_count = [i for i in range(self.seq_len) if
                         (i >= 0 and i <= 52 and i % self.filtered_seq_step_size == 0)]
        seqs_to_count = seqs_to_count[-self.filtered_seq_len:]  # filter sequence
        seqs_to_count.sort()

        for seq_ind in range(self.seq_len):
            for p in [0, 1]:
                curr_frame_img = cap.read()

                if seq_ind in seqs_to_count:
                    if self.mode == 'train':
                        img = Image.fromarray(curr_frame_img)
                        # augmentations
                        img = torchvision.transforms.functional.affine(img, angle=angle, translate=translate,
                                                                       scale=scale, shear=shear, resample=0, fillcolor=0)
                        curr_frame_img = np.array(img)
                    frames[seqs_to_count.index(seq_ind), p % trg_people_channel_num] += curr_frame_img

        if self.mode == 'train' and flip:
            label = flip_label(label)
            frames = frames[:, :, :, ::-1, :]

        if self.use_optical_flow:
            for i in range(self.filtered_seq_len - 1):
                for p in range(trg_people_channel_num):
                    flow[i, p] = self.calculate_optical_flow(frames[i, p], frames[i + 1, p])

        if self.use_pose_optical_flow:
            pose = self.poses_dict[clip_name]

            def get_y_lim(curr_poses):
                all_y = curr_poses[:, :, :, :, 1]
                all_y_non_zero = all_y[all_y > 0]
                if len(all_y_non_zero) > 0:
                    y_max = np.max(all_y_non_zero)
                    y_min = np.min(all_y_non_zero)
                else:
                    y_max = 0
                    y_min = 0
                assert not np.isinf(y_max)
                assert not np.isinf(y_min)
                return y_min, y_max

            y_min, y_max = get_y_lim(pose)
            #print('pose max/min:',y_max, y_min)

            pose[:, :, :, :, 0] = np.minimum(pose[:, :, :, :, 0] / 5.0, 1279//5)
            pose[:, :, :, :, 1] = np.minimum(pose[:, :, :, :, 1] / 2.0, 719//2)
            for i in range(self.filtered_seq_len - 1):
                for p in [0, 1]:
                    curr_flow = self.calculate_pose_optical_flow(pose[i, p], pose[i + 1, p]) #.transpose([1,0,2])
                    curr_flow = curr_flow[max(0, int(y_min / 2) - padding):min(720 // 2,int(y_max / 2) + padding), :, :]
                    #print('flow shape', curr_flow.shape)
                    curr_flow = zoom(curr_flow, np.divide((128, 256, 2), curr_flow.shape), order=0)

                    if self.mode == 'train':
                        #print(curr_flow.shape)
                        flow_img = torchvision.transforms.functional.affine(Image.fromarray(draw_hsv(curr_flow)), angle=angle, translate=translate,
                                                                       scale=scale, shear=shear, resample=0, fillcolor=0)
                        #curr_flow = affine_transform(curr_flow, affine_matrix, order=0)
                        flow_arr = np.array(flow_img)
                        curr_flow = np.stack([flow_arr[:,:,0], flow_arr[:,:,2]], axis=2)
                    flow[i, p % trg_people_channel_num] += curr_flow#.transpose([1,0,2])

        frames = frames.astype(np.float32)
        frames = frames / 255.0
        # [0, 1] => [-1, 1]
        frames = (frames * 2) - 1

        return torch.from_numpy(frames.copy()).unsqueeze(2).transpose(2, -1).squeeze(-1), \
               torch.from_numpy(flow).unsqueeze(2).transpose(2, -1).squeeze(-1), \
               label, clip_name

    def __len__(self):
        return len(self.objects)

    def calculate_optical_flow(self, prev_frame, next_frame):
        # prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        # next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        alpha = 0.012
        ratio = 1  # 0.75
        minWidth = int(180 / 4)
        nOuterFPIterations = 1
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
        img_shape = (int(256 / 4), int(128 / 4))

        prev_img = Image.fromarray(prev_frame)
        prev_img = prev_img.resize(img_shape, Image.ANTIALIAS)

        next_img = Image.fromarray(next_frame)
        next_img = next_img.resize(img_shape, Image.ANTIALIAS)

        u, v, im2W = pyflow.coarse2fine_flow(
            np.array(prev_img).astype(float) / 255.0, np.array(next_img).astype(float) / 255.0, alpha, ratio, minWidth,
            nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)

        flow = np.concatenate((u[..., None], v[..., None]), axis=2)

        # flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # flow[prev_frame_gray < 15] = 0
        flow = flow * 4.0

        flow = zoom(flow, (4, 4, 1))
        return flow.astype(np.float32)

    def get_augmentation_params(self, angle_max, translate_max, scale_range, shear_max):
        scale = uniform(0, scale_range[1] - scale_range[0]) + scale_range[0]  # 0.75-1.15
        angle = uniform(-angle_max, angle_max)
        translate_max_x = translate_max[0]
        translate_max_y = translate_max[1]
        translate = (uniform(translate_max_x[0], translate_max_x[1]), uniform(translate_max_y[0], translate_max_y[1]))
        shear = uniform(-shear_max, shear_max)
        return angle, translate, scale, shear

    def getClipInfoFromFilename(self, path):
        curr_clip_name = Path(path).stem
        curr_label = self.getLabelFromFilename(curr_clip_name)
        curr_clip_num = self.getClipNumberFromFilename(curr_clip_name)
        return curr_clip_name, curr_clip_num, curr_label

    def getLabelFromFilename(self, clip_name):
        result_letter = clip_name.split('-')[-2]
        return get_label_from_letter(result_letter)

    def getClipNumberFromFilename(self, clip_name):
        return clip_name.split('-')[-3]

    @staticmethod
    def calculate_pose_optical_flow(prev_pose, next_pose):
        flow = pose2flow(prev_pose.astype(int), next_pose.astype(int), 720//2, 1280//5)
        return flow
