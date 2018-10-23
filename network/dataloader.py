import random
import numpy as np
import torch
import torch.utils.data as torchdata
import glob
import pickle
import os


class Dataset(torchdata.Dataset):
    def __init__(self, is_train):
        descriptor_dir = '../pose_estimations'
        descriptor_files = [vid_dsc_file for vid_dsc_file in glob.glob(descriptor_dir + "/*.pickle")]

        self.objects = []

        for descriptor_file in descriptor_files:
            with (open(descriptor_file, "rb")) as file:
                ndarray_dict = pickle.load(file)
                video_dsc = np.zeros((len(ndarray_dict), 2, 17, 2, 2))

                for key in range(0, len(ndarray_dict)):
                    video_dsc[key] = ndarray_dict[str(key)]

                self.objects.append((torch.from_numpy(video_dsc), torch.from_numpy(self.getLabelFromFilename(descriptor_file))))

        if is_train:
            random.shuffle(self.objects)


    def __getitem__(self, index):
        video_dsc, label = self.objects[index]
        video_dsc = video_dsc.view(video_dsc.size(0), -1)
        return video_dsc, label


    def __len__(self):
        return len(self.objects)


    def getLabelFromFilename(self, descriptor_file):
        filename = os.path.splitext(os.path.basename(descriptor_file))[0]
        result_letter = filename.split('-')[-1]
        return self.get_result_one_hot_vector(result_letter)


    def get_result_one_hot_vector(self, letter):
        return {
            'L': np.array([1, 0, 0]),
            'R': np.array([0, 1, 0]),
            'T': np.array([0, 0, 1])
        }.get(letter)