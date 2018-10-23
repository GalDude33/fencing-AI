import random
import numpy as np
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
                self.objects.append((pickle.load(file), self.getLabelFromFilename(descriptor_file)))

        if is_train:
            random.shuffle(self.objects)


    def __getitem__(self, index):
        return self.objects[index]


    def __len__(self):
        return len(self.objects)


    def getLabelFromFilename(self, descriptor_file):
        filename = os.path.splitext(os.path.basename(descriptor_file))[0]
        result_letter = filename.split('-')[-1]
        return self.get_result_one_hot_vector(result_letter)


    def get_result_one_hot_vector(self, letter):
        return {
            'L': [1, 0, 0],
            'R': [0, 1, 0],
            'T': [0, 0, 1]
        }.get(letter)