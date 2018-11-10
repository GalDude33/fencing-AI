import glob
import os
import re
from tqdm import tqdm
import numpy as np
import collections


def getStatistics(txt_path, mode):
    video_names_to_filter = [x.rstrip() for x in open(txt_path, 'r')]
    descriptor_dir = '/media/rabkinda/DATA/fencing/FinalPoseEstimationResults/jsons*'
    descriptor_files = [vid_dsc_file for vid_dsc_file in glob.glob(descriptor_dir + "/*.json")]

    objectsMap = {}

    for descriptor_file in tqdm(descriptor_files, desc='Data loading of ' + mode):
        if any(vid_filter for vid_filter in video_names_to_filter if vid_filter in descriptor_file):

            frame_ind_str_loc = re.search("-\d{2}_\w", descriptor_file).regs[0][0] + 1
            curr_clip_name = descriptor_file[:frame_ind_str_loc - 1]
            curr_clip_name = os.path.splitext(os.path.basename(curr_clip_name))[0]

            if curr_clip_name not in objectsMap:
                objectsMap[curr_clip_name] = getLabelFromFilename(descriptor_file)

    values_count = collections.Counter(np.array([v for v in objectsMap.values()]))
    print(values_count)

    for k in values_count:
        values_count[k]=values_count[k]/len(objectsMap.values())

    print(values_count)


def getLabelFromFilename(descriptor_file):
    filename = os.path.splitext(os.path.basename(descriptor_file))[0]
    result_letter = filename.split('-')[-3]
    return result_letter


getStatistics('val.txt', 'val')
getStatistics('train.txt', 'train')
getStatistics('test.txt', 'test')
