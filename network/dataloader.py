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
from network.PoseEstimationUtils import getFencingPlayersPoseArr, normalize_point_pair_pose_arr, convert_lines_to_points
from network.utils import get_label_from_letter, flip_label


class Dataset(torchdata.Dataset):

    def __init__(self, mode, txt_path, descriptor_dir):
        print('starting dataloader init')
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

                    #if curr_label == 'T':
                    #    # if it is first clip of video, ignore it
                    #    #if int(curr_clip_num) == 0:
                    #    continue

                    if curr_clip_name not in relevant_files:
                        relevant_files[curr_clip_name] = []
                    relevant_files[curr_clip_name].append(descriptor_file)

            with Pool(max_parallel) as pool:
                inputForPool = [v for v in relevant_files.values()]
                res = list(tqdm(pool.imap(getFencingPlayersPoseArr, inputForPool, chunksize=20), total=len(inputForPool)))

            for i, curr_res in enumerate(res):
                curr_file_path = inputForPool[i][0]
                curr_clip_name, curr_clip_num, curr_label, curr_frame_ind = self.getClipInfoFromFilename(curr_file_path)

                self.objectsMap[curr_clip_name] = (curr_res, curr_label)

            for clip in self.objectsMap.keys():
                curr_clip_tuple = self.objectsMap[clip]
                self.objects.append((torch.from_numpy(curr_clip_tuple[0]), curr_clip_tuple[1], clip))

            del self.objectsMap

            with open(pickle_file, 'wb') as f:
                pickle.dump(self.objects, f)

        #self.objects = [obj for obj in self.objects if obj[1]!=2]
        self.mode = mode
        if self.mode=='train':
            random.shuffle(self.objects)

        print('finishing dataloader init')


    def __getitem__(self, index):
        video_dsc, label, base_clip_name = self.objects[index]# % len(self.objects)]

        video_dsc = video_dsc[52 - 16- 1:52]#.float()
        seq_len = video_dsc.shape[0]
        _video_dsc = []
        video_dsc_as_numpy = video_dsc.numpy()
        for seq_ind in range(seq_len):
            _video_dsc.append(convert_lines_to_points(video_dsc_as_numpy[seq_ind]))

        _video_dsc = np.array(_video_dsc)#torch.from_numpy(np.array(_video_dsc))
        flip = random.choice([0,1])
        if self.mode=='train' and flip:
            label = flip_label(label)
            _video_dsc[:,:,:,0] = 1280.0-_video_dsc[:,:,:,0]
            _video_dsc[_video_dsc == 1280.0] = 0

        y_values = _video_dsc[:, :, :, 1]#.numpy()
        y_values_bigger_than_0 = y_values[y_values > 0]

        if len(y_values_bigger_than_0)>0:
            y_min = y_values_bigger_than_0.min()
            y_max = y_values_bigger_than_0.max()
            _video_dsc[:, :, :, 1] = np.clip(a=_video_dsc[:, :, :, 1]-y_min, a_min=0, a_max=None)#torch.clamp(input=_video_dsc[:, :, :, 1]-y_min, min=0)
            _video_dsc[:, :, :, 1] = _video_dsc[:, :, :, 1]/(y_max-y_min)

        #difference of poses
        #video_dsc_mean = torch.mean(_video_dsc, dim=3)
        video_dsc_mean = _video_dsc
        video_dsc_as_diff = (video_dsc_mean[1:]-video_dsc_mean[:-1])
        video_dsc_as_diff[:,1,:,0] *= (-1)

        video_dsc_norm = normalize_point_pair_pose_arr(_video_dsc)
        #video_dsc_norm_mean_points = torch.mean(video_dsc_norm, dim=3)

        video_dcs = np.concatenate((video_dsc_as_diff, video_dsc_norm[1:]), axis=-1)#torch.cat([video_dsc_as_diff, video_dsc_norm[1:]], dim=-1)

        video_dcs = torch.from_numpy(video_dcs)
        filtered_seq_len, people_num, limbs_num, limb_feature_size = video_dcs.shape
        video_dcs = video_dcs.view(filtered_seq_len, people_num, -1)
        return video_dcs.float(), label, base_clip_name


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
