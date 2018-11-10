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

#Result:
# val:
# Counter({'T': 546, 'R': 425, 'L': 379})
# Counter({'T': 0.40444444444444444, 'R': 0.3148148148148148, 'L': 0.28074074074074074})
#
# train:
# Counter({'T': 3446, 'R': 2787, 'L': 2767})
# Counter({'T': 0.3828888888888889, 'R': 0.30966666666666665, 'L': 0.30744444444444446})
#
# test:
# Counter({'T': 947, 'R': 777, 'L': 765})
# Counter({'T': 0.38047408597830457, 'R': 0.31217356368019283, 'L': 0.3073523503415026})
