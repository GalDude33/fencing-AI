# Cuts the videos into a set of short clips where each actual hit happens. These clips are used by the data_labeller to
# label the clips where the referee had to distinguish whos priority.
import glob
import os
import ntpath
from pathlib import Path
from pylab import *
from PIL import Image
from VideoUtils import CV2VideoCapture, ffmpeg_extract_subclip, find_hit_info


videos_to_cut = glob.glob(os.getcwd() + "/precut/" + "*.mp4").__len__()
print("Cutting", videos_to_cut, "videos")

success_file_read = open("cutter_success.txt", "r")
already_processed_videos = [x.rstrip() for x in success_file_read]
success_file_read.close()

success_file = open("cutter_success.txt", "a")
error_file = open("cutter_error.txt", "w")

video_num = 0

for vid in glob.glob(os.getcwd() + "/precut/" + "*.mp4"):
    videoName = os.path.splitext(ntpath.basename(vid))[0]
    video_num += 1
    print('processing video '+ str(video_num))

    if videoName in already_processed_videos:
        continue

    print("Video:", vid)
    clips_recorded = 0

    cap = CV2VideoCapture(str(vid))
    fps = cap.get_fps()
    print("Length of Vid:", cap.__len__())

    try:
        while cap.get_position() <= cap.__len__():
            print(cap.get_position(), "big while loop", cap.__len__())
            hit_pos, next_clip_start_pos, label = find_hit_info(cap)
            if hit_pos == -1:
                break

            if hit_pos - 50>=0:
                targetname = 'clips/' + Path(vid).stem + "-" + str(clips_recorded) + "-" + label + "-" + str(hit_pos) + '.mp4'
                ffmpeg_extract_subclip(vid, t1=(hit_pos - 50)/fps, n_frames=60, targetname=targetname)
                clips_recorded += 1

            if next_clip_start_pos == -1:
                break
            cap.set_position(next_clip_start_pos)

        success_file.write(videoName + '\n')
    except:
        error_file.write(videoName + '\n')

success_file.close()
error_file.close()
print('Done!!')
