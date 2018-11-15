import glob
import os
import pickle
import random
import re
from multiprocessing.dummy import Pool
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as torchdata
from tqdm import tqdm
from network.PoseEstimationUtils import getFencingPlayersPoseArr, normalize_point_pair_pose_arr
from network.utils import get_label_from_letter


class Dataset(torchdata.Dataset):

    def __init__(self, mode, txt_path, descriptor_dir):
        video_names_to_filter = [x.rstrip() for x in open(txt_path, 'r')]
        descriptor_files = [vid_dsc_file for vid_dsc_file in glob.glob(descriptor_dir + "/*.json")]

        self.objects = []
        self.objectsMap = {}

        pickle_file = Path('network/train_val_test_splitter/'+mode+'.pickle')

        if pickle_file.is_file():
            with open(pickle_file, 'rb') as f:
                self.objects = pickle.load(f)
        else:
            max_parallel = 8
            relevant_files = {}

            for descriptor_file in descriptor_files:
                if any(vid_filter for vid_filter in video_names_to_filter if vid_filter in descriptor_file):

                    curr_clip_name, curr_clip_num, curr_label, curr_frame_ind = self.getClipInfoFromFilename(descriptor_file)

                    if curr_label == 'T':
                        # if it is first clip of video, ignore it
                        if int(curr_clip_num) == 0:
                            continue

                    if curr_clip_name not in relevant_files:
                        relevant_files[curr_clip_name] = []
                    relevant_files[curr_clip_name].append(descriptor_file)

            with Pool(max_parallel) as pool:
                inputForPool = [v for v in relevant_files.values()]
                res = list(tqdm(pool.imap(getFencingPlayersPoseArr, inputForPool, chunksize=20), total=len(inputForPool)))

            for i, curr_res in enumerate(res):
                curr_file_path = inputForPool[i][0]
                curr_clip_name, curr_clip_num, curr_label, curr_frame_ind = self.getClipInfoFromFilename(curr_file_path)

                # add normalization#TODO- deal better with normalization
                curr_res = normalize_point_pair_pose_arr(curr_res)

                self.objectsMap[curr_clip_name] = (curr_res, curr_label)

            for clip in self.objectsMap.keys():
                curr_clip_tuple = self.objectsMap[clip]
                self.objects.append((torch.from_numpy(curr_clip_tuple[0]), curr_clip_tuple[1], clip))

            del self.objectsMap

            with open(pickle_file, 'wb') as f:
                pickle.dump(self.objects, f)

        if mode=='train':
            random.shuffle(self.objects)


    def __getitem__(self, index):
        video_dsc, label, base_clip_name = self.objects[index]
        video_dsc = video_dsc.view(video_dsc.size(0), -1).float()
        return video_dsc, label, base_clip_name


    def __len__(self):
        return len(self.objects)


    def getClipInfoFromFilename(self, descriptor_file):
        frame_ind_str_loc = re.search("-\d{2}_\w", descriptor_file).regs[0][0] + 1
        frame_ind = int(descriptor_file[frame_ind_str_loc: frame_ind_str_loc + 2]) - 1
        curr_clip_name = descriptor_file[:frame_ind_str_loc - 1]
        curr_clip_name = os.path.splitext(os.path.basename(curr_clip_name))[0]
        curr_label = self.getLabelFromFilename(descriptor_file)
        curr_clip_num = self.getClipNumberFromFilename(curr_clip_name)
        return curr_clip_name, curr_clip_num, curr_label, frame_ind


    def getLabelFromFilename(self, descriptor_file):
        filename = os.path.splitext(os.path.basename(descriptor_file))[0]
        result_letter = filename.split('-')[-3]
        return get_label_from_letter(result_letter)


    def getClipNumberFromFilename(self, clip_name):
        return clip_name.split('-')[-3]
