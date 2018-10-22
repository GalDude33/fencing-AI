import torch.nn.parallel
from pytorch_Realtime_Multi_Person_Pose_Estimation.evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat
from pytorch_Realtime_Multi_Person_Pose_Estimation.network.rtpose_vgg import get_model
from pytorch_Realtime_Multi_Person_Pose_Estimation.network.post import decode_pose
from PIL import Image
import cv2


class PoseEstimatorOfficial:

    def __init__(self):
        weight_name = '../network/weight/pose_model.pth'
        self.model = get_model('vgg19')
        self.model.load_state_dict(torch.load(weight_name))
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.float()
        self.model.eval()


    def getPoseEstimationImgByPath(self, imgPath):
        img_ori = cv2.imread(imgPath) # B,G,R order
        self.getPoseEstimationImgByArr(img_ori)#[100:-100,200:-200])#TODO- DELETE!!


    def getPoseEstimationImgByArr(self, oriImg):
        multiplier = get_multiplier(oriImg)

        with torch.no_grad():
            orig_paf, orig_heat = get_outputs(
                multiplier, oriImg, self.model, 'rtpose')

            # Get results of flipped image
            swapped_img = oriImg[:, ::-1, :]
            flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
                                                    self.model, 'rtpose')

            # compute averaged heatmap and paf
            paf, heatmap = handle_paf_and_heat(
                orig_heat, flipped_heat, orig_paf, flipped_paf)

        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        canvas, to_plot, candidate, subset = decode_pose(
            oriImg, param, heatmap, paf)

        cv2.imwrite('result.png', to_plot)


    def getPoseEstimationCoordinatesByPath(self, imgPath):
        img_ori = cv2.imread(imgPath) # B,G,R order
        return self.getPoseEstimationCoordinatesByArr(img_ori)#[100:-100,200:-200])#TODO- DELETE!!


    def getPoseEstimationCoordinatesByArr(self, oriImg):
        return ''


# videoCapture = CV2VideoCapture('/media/rabkinda/Gal_Backup/fencing/fencing-AI/precut/yfTCxEAUWYI.mp4')
# videoCapture.set_position(6660)#3.42*30)
# frame = videoCapture.read()
# plt.figure(figsize=(12, 8))
# plt.imshow(frame[...,::-1])
# plt.show()

poseEstimator = PoseEstimatorOfficial()
poseEstimator.getPoseEstimationImgByPath('/media/rabkinda/Gal_Backup/fencing/fencing-AI/img2.jpg')