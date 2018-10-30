# Cuts the videos into a set of short clips where each actual hit happens. These clips are used by the data_labeller to
# label the clips where the referee had to distinguish whos priority.
import glob
import ntpath
import os
import pickle
from PoseEstimatorOfficial import PoseEstimatorOfficial
from VideoUtils import CV2VideoCapture
from PIL import Image
import cv2


poseEstimator = PoseEstimatorOfficial(weights_path='./pytorch_Realtime_Multi_Person_Pose_Estimation/network/weight/pose_model.pth')

for vid in glob.glob(os.getcwd() + "/clips*/" + "*.mp4"):
    videoName = os.path.splitext(ntpath.basename(vid))[0]

    if 'None' not in videoName:
        print('processing clip ' + videoName)
        cap = CV2VideoCapture(vid)

        clip_pose_dict = {}
        clip_paf_dict = {}
        clip_heatmap_dict = {}
        frame = cap.read()
        frame_ind = 0
        curr_output_video_path = vid.replace('clips', 'videos_with_pose').replace('clips1', 'videos_with_pose')
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
        out = cv2.VideoWriter(curr_output_video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

        while frame is not None:
            curr_frame_pose_arr, paf, heatmap = poseEstimator.getPoseEstimationCoordinatesByArr(frame)
            clip_pose_dict[str(frame_ind)] = curr_frame_pose_arr
            clip_paf_dict[str(frame_ind)] = paf
            clip_heatmap_dict[str(frame_ind)] = heatmap

            curr_frame_with_pose = poseEstimator.getPoseEstimationImgFromCoordinatesByArr(frame, curr_frame_pose_arr)
            out.write(curr_frame_with_pose)  # Write out frame to video

            print('Done with frame number ' + str(frame_ind))
            frame = cap.read()
            frame_ind += 1

        # When everything is done, release the capture
        out.release()
        cap.__del__()
        cv2.destroyAllWindows()

        output_path = vid.replace('clips', 'pose_estimations').replace('clips1', 'pose_estimations').replace('mp4', 'pickle')
        with open(output_path, 'wb') as handle:
            pickle.dump(clip_pose_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        output_path = vid.replace('clips', 'paf_estimations').replace('clips1', 'paf_estimations').replace('mp4', 'pickle')
        with open(output_path, 'wb') as handle:
            pickle.dump(clip_paf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        output_path = vid.replace('clips', 'heatmap_estimations').replace('clips1', 'heatmap_estimations').replace('mp4', 'pickle')
        with open(output_path, 'wb') as handle:
            pickle.dump(clip_heatmap_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
