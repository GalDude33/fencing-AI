# Cuts the videos into a set of short clips where each actual hit happens. These clips are used by the data_labeller to
# label the clips where the referee had to distinguish whos priority.
import glob
import os
import ntpath
from pathlib import Path
from pylab import *
from VideoUtils import CV2VideoCapture, find_hit_position, get_caption_from_video, ffmpeg_extract_subclip, find_hit_end


videos_to_cut = glob.glob(os.getcwd() + "/precut/" + "*.mp4").__len__()
print("Cutting", videos_to_cut, "videos")

already_processed_videos = [os.path.splitext(ntpath.basename(vid))[0] for vid in glob.glob(os.getcwd() + "/videos/" + "*.mp4")]
video_num = 0

for vid in glob.glob(os.getcwd() + "/precut/" + "*.mp4"):
    videoName = os.path.splitext(ntpath.basename(vid))[0]
    video_num += 1
    print('processing video '+ str(video_num))

    if np.sum([videoName in s for s in already_processed_videos])>20 or videoName == 'gir5-NQivzw':
        continue

    print("Video:", vid)
    clips_recorded = 0

    cap = CV2VideoCapture(str(vid))
    fps = cap.get_fps()
    print("Length of Vid:", cap.__len__())

    while cap.get_position() <= cap.__len__():
        print(cap.get_position(), "big while loop", cap.__len__())
        hit_pos = find_hit_position(cap)
        if hit_pos == -1:
            break

        cap.set_position(hit_pos) # more 1/2sec for caution (maybe second light is in delay)
        label = str(get_caption_from_video(cap))

        #with VideoRecorder('videos/' + Path(vid).stem + "-" + str(clips_recorded) + "-" + label + '.mp4') as vid_rec:
        ffmpeg_extract_subclip(vid, t1=(hit_pos - 50)/fps, n_frames=60, targetname='videos/' + Path(vid).stem + "-" + str(clips_recorded) + "-" + label + '.mp4')
        clips_recorded += 1
        #    vid_rec.record_video(cap, start_pos=hit_pos - 50, end_pos=hit_pos + 10, step=1)

        cap.set_position(hit_pos)
        hit_end = find_hit_end(cap)
        if hit_end == -1:
            break
        cap.set_position(hit_end)
