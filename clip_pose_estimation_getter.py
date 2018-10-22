# Cuts the videos into a set of short clips where each actual hit happens. These clips are used by the data_labeller to
# label the clips where the referee had to distinguish whos priority.
import glob
import ntpath
import os
from VideoUtils import CV2VideoCapture
from pytorch_pose_estimation.PoseEstimator import PoseEstimator


poseEstimator = PoseEstimator()

for vid in glob.glob(os.getcwd() + "/videos/" + "*.mp4"):
    videoName = os.path.splitext(ntpath.basename(vid))[0]

    if 'None' not in videoName:
        print('processing clip ' + videoName)
        cap = CV2VideoCapture(vid)

        frame = cap.read()
        while frame is not None:
            curr_frame_pose_arr = poseEstimator.getPoseEstimationCoordinatesByArr(frame)
            frame = cap.read()
