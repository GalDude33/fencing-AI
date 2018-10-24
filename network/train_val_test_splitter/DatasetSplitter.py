import random
import glob
import os
import ntpath
import re

def write_list_to_file(list, filename):
    with open(filename, 'w') as f:
        for str in list:
            f.write(str+'\n')

videos_to_split = [os.path.splitext(ntpath.basename(vid))[0] for vid in glob.glob(os.getcwd() + "/../../pose_estimations/" + "*.pickle")]
videos_to_split = [vid[:re.search("-\d{1,2}-\w", vid).regs[0][0]] for vid in videos_to_split]
videos_to_split = list(set(videos_to_split))#['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
videos_to_split.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
random.shuffle(videos_to_split) # shuffles the ordering of filenames (deterministic given the chosen seed)

split_1 = int(0.7 * len(videos_to_split))
split_2 = int(0.8 * len(videos_to_split))
train_videos = videos_to_split[:split_1]
val_videos = videos_to_split[split_1:split_2]
test_videos = videos_to_split[split_2:]

#write it to file- overwrite mode
write_list_to_file(train_videos, 'train.txt')
write_list_to_file(val_videos, 'val.txt')
write_list_to_file(test_videos, 'test.txt')
