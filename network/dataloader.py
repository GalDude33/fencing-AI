import random
import numpy as np
import torch
import torch.utils.data as torchdata
import glob
import os
import re
from network.PoseEstimationUtils import convert_points_to_lines, filterFencingPlayers, NUM_LIMBS, \
    load_people_point_pose_arr, sort_fencing_players, normalize_point_pair_pose_arr
from network.utils import get_label_from_letter
from tqdm import tqdm
from pathlib import Path
import pickle
from multiprocessing.dummy import Pool


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
            max_parallel = 16
            relevant_files = []
            frames_num = 60

            for descriptor_file in descriptor_files:
                if any(vid_filter for vid_filter in video_names_to_filter if vid_filter in descriptor_file):

                    curr_clip_name, curr_clip_num, curr_label, curr_frame_ind = self.getClipInfoFromFilename(descriptor_file)

                    if curr_label == 'T':
                        # if it is first clip of video, ignore it
                        if int(curr_clip_num) == 0:
                            continue

                    relevant_files.append(descriptor_file)

            with Pool(max_parallel) as pool:
                res = list(tqdm(pool.imap(self.getFencingPlayersPoseArr, relevant_files, chunksize=20), total=len(relevant_files)))

            for i, curr_res in enumerate(res):
                curr_file_path = relevant_files[i]
                curr_clip_name, curr_clip_num, curr_label, curr_frame_ind = self.getClipInfoFromFilename(curr_file_path)

                if curr_clip_name in self.objectsMap:
                    curr_clip_tuple = self.objectsMap[curr_clip_name]
                else:
                    curr_clip_tuple = (np.zeros((frames_num, 2, NUM_LIMBS, 2, 2), dtype=np.float32), curr_label)
                    self.objectsMap[curr_clip_name] = curr_clip_tuple

                if curr_res.shape==(2, NUM_LIMBS, 2, 2):#TODO: change it eventually!!
                    curr_clip_tuple[0][curr_frame_ind] = curr_res

            for clip in self.objectsMap.keys():
                curr_clip_tuple = self.objectsMap[clip]
                self.objects.append((torch.from_numpy(curr_clip_tuple[0]), curr_clip_tuple[1], clip))

            del self.objectsMap

            with open(pickle_file, 'wb') as f:
                pickle.dump(self.objects, f)

        if mode=='train':
            random.shuffle(self.objects)


    def getFencingPlayersPoseArr(self, descriptor_file):

        people_point_pose_arr, _ = load_people_point_pose_arr(descriptor_file)
        # filter 2 fencing players and sort them
        coords_point_pair_arr = convert_points_to_lines(people_point_pose_arr)
        fencing_players_coords = filterFencingPlayers(coords_point_pair_arr)

        if len(fencing_players_coords) == 2:#TODO: deal also with less than 2 people
            fencing_players_coords = sort_fencing_players(fencing_players_coords)

        # add normalization#TODO- deal better with normaliztion
        fencing_players_coords = normalize_point_pair_pose_arr(fencing_players_coords)

        return fencing_players_coords


    def __getitem__(self, index):
        video_dsc, label, base_clip_name = self.objects[index]
        video_dsc = video_dsc.view(video_dsc.size(0), -1)
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
