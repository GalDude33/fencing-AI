import torch.nn.parallel
from pytorch_Realtime_Multi_Person_Pose_Estimation.evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat
from pytorch_Realtime_Multi_Person_Pose_Estimation.network.rtpose_vgg import get_model
from pytorch_Realtime_Multi_Person_Pose_Estimation.network.post import decode_pose, get_pose, plot_from_pose_coords
from PIL import Image
import cv2
import numpy as np


class PoseEstimatorOfficial:

    def __init__(self, weights_path='./pytorch_Realtime_Multi_Person_Pose_Estimation/network/weight/pose_model.pth'):
        self.model = get_model('vgg19')
        self.model.load_state_dict(torch.load(weights_path))
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.float()
        self.model.eval()


    def getPoseEstimationImgByPath(self, imgPath):
        oriImg = cv2.imread(imgPath) # B,G,R order
        self.getPoseEstimationImgByArr(oriImg)


    def getPoseEstimationImgByArr(self, oriImg):
        multiplier = get_multiplier(oriImg)

        with torch.no_grad():
            orig_paf, orig_heat = get_outputs(multiplier, oriImg, self.model, 'rtpose')

            # Get results of flipped image
            swapped_img = oriImg[:, ::-1, :]
            flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img, self.model, 'rtpose')

            # compute averaged heatmap and paf
            paf, heatmap = handle_paf_and_heat(orig_heat, flipped_heat, orig_paf, flipped_paf)

        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        canvas, to_plot, candidate, subset = decode_pose(oriImg, param, heatmap, paf)

        cv2.imwrite('result.png', to_plot)


    def getPoseEstimationCoordinatesByPath(self, imgPath):
        oriImg = cv2.imread(imgPath) # B,G,R order
        return self.getPoseEstimationCoordinatesByArr(oriImg)


    def getPoseEstimationCoordinatesByArr(self, oriImg):
        multiplier = get_multiplier(oriImg)

        with torch.no_grad():
            orig_paf, orig_heat = get_outputs(multiplier, oriImg, self.model, 'rtpose')

            # Get results of flipped image
            swapped_img = oriImg[:, ::-1, :]
            flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img, self.model, 'rtpose')

            # compute averaged heatmap and paf
            paf, heatmap = handle_paf_and_heat(orig_heat, flipped_heat, orig_paf, flipped_paf)

        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        coords_arr = get_pose(oriImg, param, heatmap, paf)

        # filter 2 players which have larger distances among limbs
        fencing_players_ind = np.argsort(np.mean(np.mean(np.abs(coords_arr[:,:,0]-coords_arr[:,:,1]), 1), 1))[-2:]
        fencing_players_coords = coords_arr[fencing_players_ind]

        #verify left, right side of players
        first_player_x_mean = np.mean(fencing_players_coords[0, :, :, 0])
        second_player_x_mean = np.mean(fencing_players_coords[1, :, :, 0])

        if first_player_x_mean>second_player_x_mean:
            fencing_players_coords = fencing_players_coords[::-1]

        return fencing_players_coords


    def getPoseEstimationImgFromCoordinatesByArr(self, oriImg, coords_arr):
        canvas, to_plot = plot_from_pose_coords(oriImg, coords_arr)
        cv2.imwrite('result.png', to_plot)


# videoCapture = CV2VideoCapture('/media/rabkinda/Gal_Backup/fencing/fencing-AI/precut/yfTCxEAUWYI.mp4')
# videoCapture.set_position(6660)#3.42*30)
# frame = videoCapture.read()
# plt.figure(figsize=(12, 8))
# plt.imshow(frame[...,::-1])
# plt.show()

#poseEstimator = PoseEstimatorOfficial()
#poseEstimator.getPoseEstimationCoordinatesByPath('/media/rabkinda/Gal_Backup/fencing/fencing-AI/img2.jpg')