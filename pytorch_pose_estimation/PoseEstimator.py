import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.nn.parallel

from pytorch_pose_estimation.pose_estimation import *
from PIL import Image


class PoseEstimator:

    def __init__(self):
        self.use_gpu = True
        self.model_pose = self.loadPoseEstimationNet()
        self.scale_param = [0.5, 1.0, 1.5, 2.0]


    def loadPoseEstimationNet(self):
        state_dict = torch.load('./models/coco_pose_iter_440000.pth.tar')['state_dict']
        model_pose = get_pose_model()
        model_pose.load_state_dict(state_dict)
        model_pose.float()
        model_pose.eval()

        if self.use_gpu:
            model_pose.cuda()
            model_pose = torch.nn.DataParallel(model_pose, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        return model_pose



    def getPoseEstimationImgByPath(self, imgPath):
        img_ori = cv2.imread(imgPath) # B,G,R order
        self.getPoseEstimationImgByArr(img_ori)#[100:-100,200:-200])#TODO- DELETE!!


    def getPoseEstimationImgByArr(self, img_ori):
        paf_info, heatmap_info = get_paf_and_heatmap(self.model_pose, img_ori, self.scale_param)
        peaks = extract_heatmap_info(heatmap_info)
        sp_k, con_all = extract_paf_info(img_ori, paf_info, peaks)
        subsets, candidates = get_subsets(con_all, sp_k, peaks)
        subsets, img_points = draw_key_point(subsets, peaks, img_ori)
        img_canvas = link_key_point(img_points, candidates, subsets)

        plt.figure(figsize=(12, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(img_points[..., ::-1])

        plt.subplot(1, 2, 2)
        plt.imshow(img_canvas[..., ::-1])
        plt.show()


    def getPoseEstimationCoordinatesByPath(self, imgPath):
        img_ori = cv2.imread(imgPath) # B,G,R order
        return self.getPoseEstimationCoordinatesByArr(img_ori)#[100:-100,200:-200])#TODO- DELETE!!


    def getPoseEstimationCoordinatesByArr(self, img_ori):
        paf_info, heatmap_info = get_paf_and_heatmap(self.model_pose, img_ori, self.scale_param)
        peaks = extract_heatmap_info(heatmap_info)
        #sp_k, con_all = extract_paf_info(img_ori, paf_info, peaks)
        #subsets, candidates = get_subsets(con_all, sp_k, peaks)
        descriptor_vector = []

        for i in range(18):
            for j in range(len(peaks[i])):
                descriptor_vector.append(peaks[i][j][0:2])

        return descriptor_vector


# videoCapture = CV2VideoCapture('/media/rabkinda/Gal_Backup/fencing/fencing-AI/precut/yfTCxEAUWYI.mp4')
# videoCapture.set_position(6660)#3.42*30)
# frame = videoCapture.read()
# plt.figure(figsize=(12, 8))
# plt.imshow(frame[...,::-1])
# plt.show()

#poseEstimator = PoseEstimator()
#poseEstimator.getPoseEstimationCoordinatesByPath('/media/rabkinda/Gal_Backup/fencing/fencing-AI/img2.jpg')