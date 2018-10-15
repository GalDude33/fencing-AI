# Cuts the videos into a set of short clips where each actual hit happens. These clips are used by the data_labeller to
# label the clips where the referee had to distinguish whos priority.
import glob
import os
import subprocess as sp
import cv2
from pylab import *
from pathlib import Path

from DigitRecognizer import getDigit
from PIL import Image

# this is how long our 'recording time' will be, where we don't check for lights,
# actual recording time,
# its so long because we want to skip people testing their blades after hits
jump_length = 260

# is jump length - hide length = 'clip length'

# where we're not actually interested in keeping the frames,
# but don't want them to be seen by 'not in record mode'
hide_length = 200

videos_to_cut = glob.glob(os.getcwd() + "/precut/" + "*.mp4").__len__()
# for i in os.listdir():
#     if i.endswith(".mp4"):
#         videos_to_cut = videos_to_cut + 1

print("Cutting", videos_to_cut, "videos")


class VideoRecorder:
    def __init__(self, output_file, fps=13):
        FFMPEG_BIN = "ffmpeg"
        command = [FFMPEG_BIN,
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', '640*360',
                   '-pix_fmt', 'bgr24',
                   '-r', str(fps),
                   '-i', '-',
                   '-an',
                   '-vcodec', 'mpeg4',
                   '-b:v', '5000k',
                   output_file]

        self.proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

    def record_frame(self, frame):
        self.proc.stdin.write(frame)

    def record_video(self, video, start_pos, end_pos, step):
        if isinstance(video, str):
            cap = CV2VideoCapture(str(vid))
        elif isinstance(video, CV2VideoCapture):
            cap = video
        else:
            raise Exception("Unsupported Video Type - should be str ot CV2Recorder")

        cap.set_position(start_pos)
        for _ in range(start_pos, end_pos, step):
            frame = cap.read()
            self.record_frame(frame)

            # skip (step - 1) frames
            for _ in range(step - 1):
                _ = cap.read()

    def close(self):
        self.proc.stdin.close()
        self.proc.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class HitChecker:
    # Prelim info, FOTR light box is frame[329:334, 380:500]
    # therefore FOTL box is frame[329:334, 140:260]
    # FOTL OFF-TARGET frame[337:348, 234:250]
    # FOTR OFF-TARGET frame[337:348, 390:406]
    #      scoreleft = frame[310:325, 265:285]
    #      scoreRight = frame[310:325, 355:375]

    def __init__(self):
        self.green_box = cv2.imread("greenbox.png")
        self.red_box = cv2.imread("redbox.png")
        self.white_box = cv2.imread("whitebox.png")

    def check(self, frame):
        return (np.sum(abs(frame[337:348, 234:250].astype(int) - self.white_box.astype(int))) <= 7000) or \
               (np.sum(abs(frame[337:348, 390:406].astype(int) - self.white_box.astype(int))) <= 7000) or \
               (np.sum(abs(frame[330:334, 380:500].astype(int) - self.green_box.astype(int))) <= 40000) or \
               (np.sum(abs(frame[330:334, 140:260].astype(int) - self.red_box.astype(int))) <= 40000)


class CV2VideoCapture:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(str(video_path))
        self._cap_end_point = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self._cap_end_point

    def get_position(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def set_position(self, new_pos):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

    def read(self):
        ret, frame = self.cap.read()

        if ret == False:
            raise Exception("Couldn't read frame")

        return frame

    def __del__(self):
        self.cap.release()

    def debug(self):
        curr_pos = self.get_position()
        frame = self.read()
        Image.fromarray(frame, 'RGB').show()
        self.set_position(curr_pos)


def find_hit_position(cap: CV2VideoCapture):
    hit_checker = HitChecker()

    while cap.get_position() < cap.__len__():
        frame = cap.read()
        if hit_checker.check(frame):
            return cap.get_position()

    return -1


def find_hit_end(cap: CV2VideoCapture):
    hit_checker = HitChecker()

    while cap.get_position() < cap.__len__():
        frame = cap.read()
        if not hit_checker.check(frame):
            return cap.get_position()

    return -1


already_processed = 0
for vid in glob.glob(os.getcwd() + "/precut/" + "*.mp4"):
    if int(Path(vid).stem) >= already_processed:
        print("Video:", vid)
        clips_recorded = 0

        cap = CV2VideoCapture(str(vid))
        print("Length of Vid:", cap.__len__())

        while cap.get_position() <= cap.__len__():
            print(cap.get_position(), "big while loop", cap.__len__())

            hit_pos = find_hit_position(cap)
            if hit_pos == -1:
                break

            with VideoRecorder('videos/' + Path(vid).stem + "-" + str(clips_recorded) + '.mp4') as vid_rec:
                clips_recorded += 1
                vid_rec.record_video(cap, start_pos=hit_pos - 50, end_pos=hit_pos+10, step=1)

            cap.set_position(hit_pos)
            hit_end = find_hit_end(cap)
            if hit_end == -1:
                break

            cap.set_position(hit_end)