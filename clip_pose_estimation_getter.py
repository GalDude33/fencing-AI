# Cuts the videos into a set of short clips where each actual hit happens. These clips are used by the data_labeller to
# label the clips where the referee had to distinguish whos priority.
import glob
import ntpath
import os
import pickle
from PoseEstimatorOfficial import PoseEstimatorOfficial
from VideoUtils import CV2VideoCapture
from PIL import Image


poseEstimator = PoseEstimatorOfficial(weights_path='./pytorch_Realtime_Multi_Person_Pose_Estimation/network/weight/pose_model.pth')

for vid in glob.glob(os.getcwd() + "/videos/" + "*.mp4"):
    videoName = os.path.splitext(ntpath.basename(vid))[0]

    if 'None' not in videoName:
        print('processing clip ' + videoName)
        cap = CV2VideoCapture(vid)

        clip_pose_dict = {}
        frame = cap.read()
        frame_ind = 0

        while frame is not None:
            curr_frame_pose_arr = poseEstimator.getPoseEstimationCoordinatesByArr(frame)
            clip_pose_dict[str(frame_ind)] = curr_frame_pose_arr
            frame = cap.read()
            frame_ind += 1

        output_path = vid.replace('videos', 'pose_estimations').replace('mp4', 'pickle')
        with open(output_path, 'wb') as handle:
            pickle.dump(clip_pose_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
