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
from skimage import transform
import ntpath


videos_to_cut = glob.glob(os.getcwd() + "/precut/" + "*.mp4").__len__()
print("Cutting", videos_to_cut, "videos")


def ffmpeg_extract_subclip(filename, t1, n_frames, targetname=None):
    """ Makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    cmd = ["ffmpeg", "-y",
           "-i", filename,
           "-strict -2",
           "-ss", "%0.3f" % t1,
           "-frames:v", "%d" % n_frames,
           targetname]

    # with open(os.devnull, "w") as f:
    #     sp.Popen(cmd, stdout=f, stderr=f)

    cmd = sp.Popen(' '.join(cmd), stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    out, err = cmd.communicate()

    if cmd.returncode == 0:
        print("Job done.")
    else:
        print("ERROR")
        print(err)


class VideoRecorder:
    def __init__(self, output_file, fps=13):
        FFMPEG_BIN = "ffmpeg"
        command = [FFMPEG_BIN,
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', '1280*720', #'640*360',
                   '-pix_fmt', 'bgr24',
                   '-r', str(fps),
                   '-i', '-',
                   '-an',
                   '-vcodec', 'mpeg4',
                   #'-b:v', '2M', #'5000k',
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
            if step > 1:
                cap.skip(step - 1)

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

    #def __init__(self):
    def load_and_resize(img_path, factor = 2):
        box = cv2.imread(img_path)
        box = (transform.resize(box, np.multiply(box.shape[:-1], factor)) * 255).astype(np.uint8)
        return box

    green_box = load_and_resize("greenbox.png")
    red_box = load_and_resize("redbox.png")
    white_box = load_and_resize("whitebox.png")

    @classmethod
    def check_green(cls, frame):
        return np.sum(abs(frame[330*2:334*2, 380*2:500*2].astype(int) - cls.green_box.astype(int))) <= 4*40000

    @classmethod
    def check_red(cls, frame):
        return np.sum(abs(frame[330*2:334*2, 140*2:260*2].astype(int) - cls.red_box.astype(int))) <= 4*40000

    @classmethod
    def check_left_white(cls, frame):
        return np.sum(abs(frame[337*2:348*2, 234*2:250*2].astype(int) - cls.white_box.astype(int))) <= 4*7000

    @classmethod
    def check_right_white(cls, frame):
        return np.sum(abs(frame[337*2:348*2, 390*2:406*2].astype(int) - cls.white_box.astype(int))) <= 4*7000

    @classmethod
    def check(cls, frame):
        return cls.check_left_white(frame) or \
               cls.check_right_white(frame) or \
               cls.check_green(frame) or \
               cls.check_red(frame)


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

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def read(self):
        ret, frame = self.cap.read()

        if ret == False:
            if self.get_position() >= self.__len__() - 1:
                return None
            else:
                raise Exception("Couldn't read frame")

        return frame

    def skip(self, n):
        self.set_position(self.get_position()+n)
        return self

    def __del__(self):
        self.cap.release()

    def debug(self):
        curr_pos = self.get_position()
        frame = self.read()
        Image.fromarray(frame, 'RGB').show()
        self.set_position(curr_pos)


def find_hit_position(cap: CV2VideoCapture, step=15):
    while cap.get_position() < cap.__len__():
        frame = cap.read()
        if HitChecker.check(frame):
            return cap.get_position()
        cap.skip(step - 1)

    return -1


def find_hit_end(cap: CV2VideoCapture, step=15):
    while cap.get_position() < cap.__len__():
        frame = cap.read()
        if not HitChecker.check(frame):
            return cap.get_position()
        cap.skip(step - 1)

    return -1


def get_scores(frame):
    left_score = getDigit(frame[309*2:325*2, 265*2:285*2])
    right_score = getDigit(frame[309*2:325*2, 355*2:375*2])
    # print(left_score, right_score)
    return left_score, right_score


def find_score_change(cap: CV2VideoCapture, step=15):
    frame = cap.read()
    orig_left_score, orig_right_score = get_scores(frame)
    while cap.get_position() < cap.__len__():
        try:
            frame = cap.read()
            left_score, right_score = get_scores(frame)
            if left_score != orig_left_score or right_score != orig_right_score:
                return cap.get_position(), left_score, right_score
        except ValueError as e:
            print(e)
            pass
        cap.skip(step - 1)

    return -1, None, None


def caption(hit_type, left, right, update_left, update_right):
    caption = "None"
    if hit_type == "On-On":
        if update_left - left == 1 and update_right - right == 0:
            caption = "L"
        if update_left - left == 0 and update_right - right == 1:
            caption = "R"
        if update_left - left == 0 and update_right - right == 0:
            caption = "T"
    if hit_type == "On-Off":
        if update_left - left == 1 and update_right - right == 0:
            caption = "L"
        if update_left - left == 0 and update_right - right == 0:
            caption = "R"
    if hit_type == "Off-On":
        if update_left - left == 0 and update_right - right == 1:
            caption = "R"
        if update_left - left == 0 and update_right - right == 0:
            caption = "L"

    return caption


def check_lights(frame):
    # returns a string, either On-On, On-Off, Off-On, Off-Off, On-No, No-On, Off-No, No-Off

    # red is on the left, green on the right
    string = ""
    # check for left on target light
    if HitChecker.check_red(frame):
        string = string + "On"
    # check for left off target light
    elif HitChecker.check_left_white(frame):
        string = string + "Off"
    else:
        string = string + "No"

    # check for right off target light
    string = string + "-"
    if HitChecker.check_green(frame):
        string = string + "On"
    # check for right on target light
    elif HitChecker.check_right_white(frame):
        string = string + "Off"
    else:
        string = string + "No"

    return string

def get_caption_from_video(cap):
    orig_pos = cap.get_position()
    orig_frame = cap.skip(30).read()

    # find next hit
    hit_end_pos = find_hit_end(cap)
    cap.set_position(hit_end_pos+1)
    next_hit_pos = find_hit_position(cap)

    # find score change
    cap.set_position(orig_pos)
    prev_l_score, prev_r_score = get_scores(orig_frame)
    score_changed_pos, l_score, r_score = find_score_change(cap, 15)

    if score_changed_pos == -1 or score_changed_pos > next_hit_pos: # score didn't changed
        l_score, r_score = prev_l_score, prev_r_score

    if l_score - prev_l_score > 1 or r_score - prev_r_score > 1:
        raise Exception("Missed score")

    cap.set_position(orig_pos)
    hit_type = check_lights(orig_frame)
    if hit_type=="No-No":
        raise Exception("got No-No")

    if hit_type == "On-On" or hit_type == "On-Off" or hit_type == "Off-On":
        return caption(hit_type, prev_l_score, prev_r_score, l_score, r_score)
    else:
        return None


# already_processed = 0
# for vid in sorted(glob.glob(os.getcwd() + "/precut/" + "*.mp4"), key=lambda x: int(Path(x).stem)):
#     if int(Path(vid).stem) >= already_processed:

already_processed_videos = [os.path.splitext(ntpath.basename(vid))[0] for vid in glob.glob(os.getcwd() + "/videos/" + "*.mp4")]
video_num = 0

for vid in glob.glob(os.getcwd() + "/precut/" + "*.mp4"):
    videoName = os.path.splitext(ntpath.basename(vid))[0]
    video_num += 1
    print('processing video '+ str(video_num))

    if np.sum([videoName in s for s in already_processed_videos])>20 and videoName == 'gir5-NQivzw':
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
