import itertools
import random
import glob
import os
import ntpath
import re
from network.utils import getLabelFromFilename


def write_list_to_file(list, filename):
    with open(filename, 'w') as f:
        for str in list:
            f.write(str + '\n')

videos_to_split = [os.path.splitext(ntpath.basename(vid))[0] for vid in
                   glob.glob('/media/rabkinda/DATA/fencing/FinalPoseEstimationResults/jsons*/*.json')]
videos_to_split = [vid[:re.search("-\d{1,2}_keypoints", vid).regs[0][0]] for vid in videos_to_split]
videos_to_split = list(set(videos_to_split))
videos_to_split.sort()  # make sure that the filenames have a fixed order before shuffling

def get_video_by_label(videos_to_split, label):
    return list(filter(lambda x: getLabelFromFilename(x) == label, videos_to_split))

L_vids = get_video_by_label(videos_to_split, 'L')
R_vids = get_video_by_label(videos_to_split, 'R')
T_vids = get_video_by_label(videos_to_split, 'T')

all_vids = [L_vids, R_vids, T_vids]

for vid in all_vids:
    random.seed(230)
    vid.sort()
    random.shuffle(vid)  # shuffles the ordering of filenames (deterministic given the chosen seed)

train_split = 0.8
val_split = 0.9

train_videos = itertools.chain.from_iterable(vids[:int(train_split * len(vids))] for vids in all_vids)
val_videos = itertools.chain.from_iterable(vids[int(train_split * len(vids)):int(val_split * len(vids))] for vids in all_vids)
test_videos = itertools.chain.from_iterable(vids[int(val_split * len(vids)):] for vids in all_vids)

# write it to file- overwrite mode
write_list_to_file(train_videos, 'train.txt')
write_list_to_file(val_videos, 'val.txt')
write_list_to_file(test_videos, 'test.txt')
