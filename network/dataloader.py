import random
import numpy as np
import torch
import torch.utils.data as torchdata
import glob
import pickle
import os
import ntpath
import json
import re
from network.utils import get_label_from_letter


class Dataset(torchdata.Dataset):
    def __init__(self, is_train, txt_path):
        video_names_to_filter = [x.rstrip() for x in open(txt_path, 'r')]
        descriptor_dir = '/media/rabkinda/DATA/fencing/FinalPoseEstimationResults/jsons*'
        descriptor_files = [vid_dsc_file for vid_dsc_file in glob.glob(descriptor_dir + "/*.json")]

        self.objects = []
        self.objectsMap = {}

        for descriptor_file in descriptor_files:
            if any(vid_filter for vid_filter in video_names_to_filter if vid_filter in descriptor_file):
                with (open(descriptor_file, "rb")) as file:
                    people_dict = json.load(file)['people']
                    people_pose_arr = np.array([np.array(p['pose_keypoints_2d']) for p in people_dict])

                    #filter 2 fencing players
                    people_pose_arr = people_pose_arr[:2]#TODO: change it!!

                    frame_ind_str_loc = re.search("-\d{2}_\w", descriptor_file).regs[0][0] + 1
                    frame_ind = int(descriptor_file[frame_ind_str_loc: frame_ind_str_loc + 2])-1

                    curr_clip_name = descriptor_file[:frame_ind_str_loc - 1]
                    curr_clip_name = os.path.splitext(os.path.basename(curr_clip_name))[0]

                    if curr_clip_name in self.objectsMap:
                        curr_clip_tuple = self.objectsMap[curr_clip_name]
                    else:
                        curr_clip_tuple = (np.zeros((60, 2, 75), dtype=np.float32), self.getLabelFromFilename(descriptor_file))
                        self.objectsMap[curr_clip_name] = curr_clip_tuple

                    curr_clip_tuple[0][frame_ind] = people_pose_arr

        for clip in self.objectsMap.keys():
            curr_clip_tuple = self.objectsMap[clip]
            self.objects.append((torch.from_numpy(curr_clip_tuple[0]), curr_clip_tuple[1], clip))

        del self.objectsMap

        if is_train:
            random.shuffle(self.objects)


    def __getitem__(self, index):
        video_dsc, label, base_clip_name = self.objects[index]
        video_dsc = video_dsc.view(video_dsc.size(0), -1)
        return video_dsc, label, base_clip_name


    def __len__(self):
        return len(self.objects)


    def getLabelFromFilename(self, descriptor_file):
        filename = os.path.splitext(os.path.basename(descriptor_file))[0]
        result_letter = filename.split('-')[-3]
        return get_label_from_letter(result_letter)
