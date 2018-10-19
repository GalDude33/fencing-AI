import os
import cv2
import sys
import math
import time
# import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.autograd import Variable

from utils import *
from pose_estimation import *
from scipy.ndimage.filters import gaussian_filter

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'


use_gpu = True

test_image = './test.jpg'
img_ori = cv2.imread(test_image) # B,G,R order

# display the validation pics
plt.figure(figsize=(12, 8))
plt.imshow(img_ori[...,::-1])

state_dict = torch.load('./models/coco_pose_iter_440000.pth.tar')['state_dict']

model_pose = get_pose_model()
model_pose.load_state_dict(state_dict)
model_pose.float()
model_pose.eval()

if use_gpu:
    model_pose.cuda()
    model_pose = torch.nn.DataParallel(model_pose, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

scale_param = [0.5, 1.0, 1.5, 2.0]
paf_info, heatmap_info = get_paf_and_heatmap(model_pose, img_ori, scale_param)

peaks = extract_heatmap_info(heatmap_info)

sp_k, con_all = extract_paf_info(img_ori, paf_info, peaks)

subsets, candidates = get_subsets(con_all, sp_k, peaks)

subsets, img_points = draw_key_point(subsets, peaks, img_ori)
img_canvas = link_key_point(img_points, candidates, subsets)

# cv2.imwrite('result.png', img_canvas)

plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.imshow(img_points[...,::-1])

plt.subplot(1, 2, 2)
plt.imshow(img_canvas[...,::-1])