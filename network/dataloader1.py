import random
import numpy as np
import torch
import torch.utils.data as torchdata
import glob
import os
import json
import re
from network.utils import get_label_from_letter
from tqdm import tqdm
from pathlib import Path
import pickle


class Dataset(torchdata.Dataset):
    def __init__(self, mode, txt_path):
        video_names_to_filter = [x.rstrip() for x in open(txt_path, 'r')]
        descriptor_dir = '/media/rabkinda/DATA/fencing/FinalPoseEstimationResults/jsons*'
        descriptor_files = [vid_dsc_file for vid_dsc_file in glob.glob(descriptor_dir + "/*.json")]

        self.objects = []
        self.objectsMap = {}

        pickle_file = Path('train_val_test_splitter/'+mode+'.pickle')
        if pickle_file.is_file():
            with open(pickle_file, 'rb') as f:
                self.objects = pickle.load(f)
        else:
            for descriptor_file in tqdm(descriptor_files, desc='Data loading of '+mode):
                if any(vid_filter for vid_filter in video_names_to_filter if vid_filter in descriptor_file):

                    frame_ind_str_loc = re.search("-\d{2}_\w", descriptor_file).regs[0][0] + 1
                    frame_ind = int(descriptor_file[frame_ind_str_loc: frame_ind_str_loc + 2]) - 1

                    curr_clip_name = descriptor_file[:frame_ind_str_loc - 1]
                    curr_clip_name = os.path.splitext(os.path.basename(curr_clip_name))[0]

                    curr_label = self.getLabelFromFilename(descriptor_file)

                    if curr_label == 'T':
                        # if it is first clip of video, ignore it
                        curr_clip_num = self.getClipNumberFromFilename(curr_clip_name)
                        if int(curr_clip_num) == 0:
                            continue

                    with (open(descriptor_file, "rb")) as file:
                        people_dict = json.load(file)['people']
                        people_pose_arr = np.array([np.array(p['pose_keypoints_2d']) for p in people_dict])

                        people_pose_arr = people_pose_arr.reshape(people_pose_arr.shape[0], 25, 3)
                        people_pose_confidence_arr = people_pose_arr[:, :, -1]
                        people_pose_arr = people_pose_arr[:, :, :2]
                        #filter 2 fencing players and sort them
                        fencing_players_coords = self.filterFencingPlayers(people_pose_arr)

                        if len(fencing_players_coords)==2:
                            # verify left, right side of players
                            first_player_x_mean = np.mean(fencing_players_coords[0, :, :, 0])
                            second_player_x_mean = np.mean(fencing_players_coords[1, :, :, 0])

                            if first_player_x_mean > second_player_x_mean:
                                fencing_players_coords = fencing_players_coords[::-1]

                            #add normalization
                            fencing_players_coords[:, :, :, 0] = fencing_players_coords[:, :, :, 0] / 1280.0
                            fencing_players_coords[:, :, :, 1] = fencing_players_coords[:, :, :, 1] / 720.0

                        if curr_clip_name in self.objectsMap:
                            curr_clip_tuple = self.objectsMap[curr_clip_name]
                        else:
                            curr_clip_tuple = (np.zeros((60, 2, 26, 2, 2), dtype=np.float32), curr_label)
                            self.objectsMap[curr_clip_name] = curr_clip_tuple

                        if people_pose_arr.shape==(2, 26, 2, 2):
                            curr_clip_tuple[0][frame_ind] = fencing_players_coords
                        # else:
                        #     print('not all players found in clip: '+curr_clip_name+' in frame number '+str(frame_ind))

            for clip in self.objectsMap.keys():
                curr_clip_tuple = self.objectsMap[clip]
                self.objects.append((torch.from_numpy(curr_clip_tuple[0]), curr_clip_tuple[1], clip))

            del self.objectsMap

            with open(pickle_file, 'wb') as f:
                pickle.dump(self.objects, f)

        for obj in self.objects:
            curr_tensor = obj[0]
            curr_tensor = curr_tensor[:50]

        if mode=='train':
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


    def getClipNumberFromFilename(self, clip_name):
        return clip_name.split('-')[-3]


    def filterFencingPlayers(self, coords_arr):
        posePartPairs = self.getPosePartPairs()
        coords_pair_arr = np.zeros((coords_arr.shape[0], 26, 2, 2))

        for i, pair in enumerate(posePartPairs):
            coords_pair_arr[:, i] = np.append(arr=coords_arr[:, pair[0], :], values=coords_arr[:, pair[1], :], axis=1)\
                .reshape(coords_arr.shape[0], 2, 2)

        for p in range(coords_arr.shape[0]):
            for l in range(26):
                curr_limb = coords_pair_arr[p, l]
                if (curr_limb[0]==np.array([0, 0])).min() or (curr_limb[1]==np.array([0, 0])).min():
                    coords_pair_arr[p, l] = np.array([[-1., -1.], [-1., -1.]])

        fencing_players_ind = np.argsort(np.sum(np.sum(np.abs(coords_pair_arr[:,:,0]-coords_pair_arr[:,:,1]), 1), 1))[-3:]
        curr_fencing_players_coords = coords_pair_arr[fencing_players_ind]
        fencing_players_ind = np.argsort(np.array([np.count_nonzero(p_arr == -1.) for p_arr in curr_fencing_players_coords]))[:2]
        curr_fencing_players_coords = curr_fencing_players_coords[fencing_players_ind]
        return curr_fencing_players_coords


    def getPosePartPairs(self):
        posePartPairs = np.array([1, 8, 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 8,
         12, 12, 13, 13, 14, 1, 0, 0, 15, 15, 17, 0, 16, 16, 18, 2, 17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24])
        posePartPairs = posePartPairs.reshape(-1, 2)
        return posePartPairs
