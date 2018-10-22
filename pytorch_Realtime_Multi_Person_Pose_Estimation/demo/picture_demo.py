import os
import re
import sys
#sys.path.insert(0, './pytorch_Realtime_Multi_Person_Pose_Estimation')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter
from pytorch_Realtime_Multi_Person_Pose_Estimation.network.rtpose_vgg import get_model
from pytorch_Realtime_Multi_Person_Pose_Estimation.network.post import decode_pose
from pytorch_Realtime_Multi_Person_Pose_Estimation.training.datasets.coco_data.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from pytorch_Realtime_Multi_Person_Pose_Estimation.network import im_transform
from pytorch_Realtime_Multi_Person_Pose_Estimation.evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat


weight_name = '../network/weight/pose_model.pth'

model = get_model('vgg19')     
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()



test_image = '/media/rabkinda/Gal_Backup/fencing/fencing-AI/img2.jpg'#'./readme/ski.jpg'
oriImg = cv2.imread(test_image) # B,G,R order
shape_dst = np.min(oriImg.shape[0:2])

# Get results of original image
multiplier = get_multiplier(oriImg)

with torch.no_grad():
    orig_paf, orig_heat = get_outputs(
        multiplier, oriImg, model,  'rtpose')
          
    # Get results of flipped image
    swapped_img = oriImg[:, ::-1, :]
    flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
                                            model, 'rtpose')

    # compute averaged heatmap and paf
    paf, heatmap = handle_paf_and_heat(
        orig_heat, flipped_heat, orig_paf, flipped_paf)
            
param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
canvas, to_plot, candidate, subset = decode_pose(
    oriImg, param, heatmap, paf)
 
cv2.imwrite('result.png',to_plot)   

