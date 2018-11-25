import glob
import os
import re
from tqdm import tqdm
import numpy as np
import collections
from network.utils import getLabelFromFilename


# def getStatistics(txt_path, mode):
#     video_names_to_filter = [x.rstrip() for x in open(txt_path, 'r')]
#     descriptor_dir = '/media/rabkinda/DATA/fencing/FinalPoseEstimationResults/jsons*'
#     descriptor_files = [vid_dsc_file for vid_dsc_file in glob.glob(descriptor_dir + "/*.json")]
#
#     objectsMap = {}
#
#     for descriptor_file in tqdm(descriptor_files, desc='Data loading of ' + mode):
#         if any(vid_filter for vid_filter in video_names_to_filter if vid_filter in descriptor_file):
#
#             frame_ind_str_loc = re.search("-\d{2}_\w", descriptor_file).regs[0][0] + 1
#             curr_clip_name = descriptor_file[:frame_ind_str_loc - 1]
#             curr_clip_name = os.path.splitext(os.path.basename(curr_clip_name))[0]
#
#             if curr_clip_name not in objectsMap:
#                 label = getLabelFromFilename(descriptor_file)
#
#                 #if label=='T':
#                     #if it is first clip of video, ignore it
#                     #clip_num = getClipNumberFromFilename(curr_clip_name)
#                     #if int(clip_num)==0:
#                     #    continue
#
#                 objectsMap[curr_clip_name] = label
#
#     values_count = collections.Counter(np.array([v for v in objectsMap.values()]))
#     print(values_count)
#
#     for k in values_count:
#         values_count[k]=values_count[k]/len(objectsMap.values())
#
#     print(values_count)


def getStatistics(txt_path, mode):
    print('Statistics of '+mode+':')
    clips_names_to_filter = [x.rstrip() for x in open(txt_path, 'r')]

    label_map = {'L':0, 'R':0, 'T':0}

    for clip_name in clips_names_to_filter:
        label_map[getLabelFromFilename(clip_name)]+=1

    print(label_map)

    for k in label_map:
        label_map[k]=label_map[k]/len(clips_names_to_filter)

    print(label_map)
    print('\n')


# def getClipNumberFromFilename(clip_name):
#     return clip_name.split('-')[-3]


getStatistics('val.txt', 'val')
getStatistics('train.txt', 'train')
getStatistics('test.txt', 'test')

#Result:
# val:
# Counter({'T': 503, 'R': 425, 'L': 379})
# Counter({'T': 0.38485080336648814, 'R': 0.32517214996174443, 'L': 0.2899770466717674})
#
# train:
# Counter({'T': 3134, 'R': 2787, 'L': 2767})
# Counter({'T': 0.36072744014732966, 'R': 0.32078729281767954, 'L': 0.3184852670349908})
#
# test:
# Counter({'T': 858, 'R': 777, 'L': 765})
# Counter({'T': 0.3575, 'R': 0.32375, 'L': 0.31875})
