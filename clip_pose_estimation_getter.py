# Cuts the videos into a set of short clips where each actual hit happens. These clips are used by the data_labeller to
# label the clips where the referee had to distinguish whos priority.
import glob
import ntpath
import os
import pickle
from PoseEstimatorOfficial import PoseEstimatorOfficial
from VideoUtils import CV2VideoCapture
from PIL import Image
from datetime import datetime
import cv2
import scipy
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool


poseEstimator = PoseEstimatorOfficial(weights_path='./pytorch_Realtime_Multi_Person_Pose_Estimation/network/weight/pose_model.pth')
batch_size = 10
POOL_SIZE = 6


def saveCoordinates(input):
    img, paf, heatmap = input
    #print('got here!!')
    curr_fencing_players_coords = poseEstimator.getPoseEstimationCoordinates(img, paf, heatmap)
    curr_frame_with_pose = poseEstimator.getPoseEstimationImgFromCoordinatesByArr(img, curr_fencing_players_coords)
    return curr_fencing_players_coords, curr_frame_with_pose
    #clip_pose_dict[str(frame_ind)] = curr_fencing_players_coords


for vid in glob.glob(os.getcwd() + "/clips2/" + "*.mp4"):
    videoName = os.path.splitext(ntpath.basename(vid))[0]

    if 'None' not in videoName:
        videoStartTime = datetime.now()
        pool = ThreadPool(processes=POOL_SIZE)
        print('processing clip ' + videoName)
        cap = CV2VideoCapture(vid)

        clip_pose_dict = {}
        #clip_paf_dict = {}
        #clip_heatmap_dict = {}
        frame = cap.read()
        #frame_ind = 0
        curr_output_video_path = vid.replace('clips2', 'videos_with_pose').replace('clips2', 'videos_with_pose')
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
        img_shape = (640, 360)#(int(frame.shape[1]/2), int(frame.shape[0]/2))
        #print(img_shape)
        out = cv2.VideoWriter(curr_output_video_path, fourcc, 20.0, img_shape)
        frame_arr = np.zeros((50, img_shape[1], img_shape[0], 3), dtype=np.uint8)
        for i in range(50):
            frame = cap.read()
            frame_arr[i] = scipy.misc.imresize(frame, (img_shape[1], img_shape[0]))

        results = {}
        for i in range(0, 50, batch_size):
            #frame = scipy.misc.imresize(frame, (img_shape[1], img_shape[0]))
            frames_batch = frame_arr[i:i+batch_size]
            #curr_frames_pose_arr, _, _ = poseEstimator.getPoseEstimationCoordinatesByArr(frames_batch)
            paf, heatmap = poseEstimator.getPoseEstimationNetworkResultByArr(frames_batch)

            #for j in range(batch_size):
            arguments = [(frames_batch[j], paf[j], heatmap[j]) for j in range(batch_size)]
            results[str(i)] = pool.imap(saveCoordinates , arguments)
            #curr_frame_with_pose = poseEstimator.getPoseEstimationImgFromCoordinatesByArr(frames_batch[j], curr_fencing_players_coords)
            #out.write(curr_frame_with_pose)  # Write out frame to video
            #frame_ind += 1

            #clip_paf_dict[str(frame_ind)] = paf
            #clip_heatmap_dict[str(frame_ind)] = heatmap

            #curr_frame_with_pose = poseEstimator.getPoseEstimationImgFromCoordinatesByArr(frame, curr_frame_pose_arr)
            #out.write(curr_frame_with_pose)  # Write out frame to video

            print('Done with frame number ' + str(i+batch_size))
            #frame = cap.read()

        pool.close()
        pool.join()

        for i, result in results.items():
            for j in range(batch_size):
                curr_res = result.next()
                frame_ind = int(i)+j
                clip_pose_dict[str(frame_ind)] = curr_res[0]
                out.write(curr_res[1])  # Write out frame to video

        # When everything is done, release the capture
        out.release()
        cap.__del__()
        cv2.destroyAllWindows()

        output_path = vid.replace('clips2', 'pose_estimations').replace('clips2', 'pose_estimations').replace('mp4', 'pickle')
        with open(output_path, 'wb') as handle:
            pickle.dump(clip_pose_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # output_path = vid.replace('clips', 'paf_estimations').replace('clips1', 'paf_estimations').replace('mp4', 'pickle')
        # with open(output_path, 'wb') as handle:
        #     pickle.dump(clip_paf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # output_path = vid.replace('clips', 'heatmap_estimations').replace('clips1', 'heatmap_estimations').replace('mp4', 'pickle')
        # with open(output_path, 'wb') as handle:
        #     pickle.dump(clip_heatmap_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('clip processing time = '+str(datetime.now()-videoStartTime))
