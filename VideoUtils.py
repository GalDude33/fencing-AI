import subprocess as sp
import cv2
from PIL import Image
from pylab import *
from skimage import transform
from DigitRecognizer import getDigit


def ffmpeg_extract_subclip(filename, t1, n_frames, targetname=None):
    """ Makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    cmd = ["ffmpeg", "-y",
           "-ss", "%0.3f" % t1,
           "-i", filename,
           "-strict -2",
           "-frames:v", "%d" % n_frames,
           targetname]

    # with open(os.devnull, "w") as f:
    #     sp.Popen(cmd, stdout=f, stderr=f)

    cmd = sp.Popen(' '.join(cmd), stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    # out, err = cmd.communicate()
    #
    # if cmd.returncode == 0:
    #     print("Job done.")
    # else:
    #     print("ERROR")
    #     print(err)

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

    green_box = load_and_resize('greenbox.png')
    red_box = load_and_resize('redbox.png')
    white_box = load_and_resize('whitebox.png')

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
    def check(cls, frame, including_white):
        light_on_list = []

        if including_white and cls.check_left_white(frame):
            light_on_list.append('left_white')
        if including_white and cls.check_right_white(frame):
            light_on_list.append('right_white')
        if cls.check_green(frame):
            light_on_list.append('green')
        if cls.check_red(frame):
            light_on_list.append('red')

        light_on = size(light_on_list)>0
        return light_on, light_on_list

    @classmethod
    #returns 0 if first list is greater, 1 if second list is greater, and 2 if they equal
    def get_max_lights_list(cls, first_light_on_list, second_light_on_list):
        if set(first_light_on_list)==set(second_light_on_list):
            return 2

        def filter_white(list):
            return [e for e in list if 'white' not in e]

        def get_more_elements_list(list1, list2):
            len_diff = len(list1)-len(list2)
            if len_diff>0:
                return 0
            elif len_diff<0:
                return 1
            else:
                return 2

        first_list_without_white = filter_white(first_light_on_list)
        second_list_without_white = filter_white(second_light_on_list)

        res = get_more_elements_list(first_list_without_white, second_list_without_white)
        if res==2:
            res = get_more_elements_list(first_light_on_list, second_light_on_list)
        return res


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
        for _ in range(n):
            self.read() #set_position(self.get_position()+n)
        return self

    def __del__(self):
        self.cap.release()

    def debug(self):
        curr_pos = self.get_position()
        frame = self.read()
        Image.fromarray(frame, 'RGB').show()
        self.set_position(curr_pos)


def find_hit_info(cap):
    hit_pos, next_clip_start_pos, label = -1, -1, "None"

    first_light_on_pos, _ = find_hit_position(cap)
    if first_light_on_pos==-1:
        return hit_pos, next_clip_start_pos, label

    prev_score_frame_pos = first_light_on_pos - 1 if first_light_on_pos - 1>=0 else first_light_on_pos
    cap.set_position(prev_score_frame_pos)
    frame = cap.read()
    pre_left_score, pre_right_score = get_scores(frame)
    if pre_left_score==15 or pre_right_score==15:
        return hit_pos, next_clip_start_pos, label

    #go from first_light_on_pos to first_light_off_pos, and save the frame which has bigger number of lights on
    hit_pos, _, hit_pos_frame = find_max_hit_lights(cap, first_light_on_pos)

    hit_type = check_lights(hit_pos_frame)
    if hit_type == "No-No":
        raise Exception("got No-No")

    next_clip_start_pos, post_left_score, post_right_score = find_post_hit_score(cap)

    if post_left_score - pre_left_score > 1 or post_right_score - pre_right_score > 1:
        raise Exception("Missed score")

    if hit_type == "On-On" or hit_type == "On-Off" or hit_type == "Off-On":
        label = caption(hit_type, pre_left_score, pre_right_score, post_left_score, post_right_score)

    return hit_pos, next_clip_start_pos, label


def find_max_hit_lights(cap: CV2VideoCapture, first_light_on_pos):
    hit_pos, hit_light_on_list, hit_pos_frame = first_light_on_pos, [], None

    while cap.get_position() < cap.__len__():
        frame = cap.read()
        curr_light_on, curr_light_on_list = HitChecker.check(frame, including_white=True)

        if HitChecker.get_max_lights_list(curr_light_on_list, hit_light_on_list)==0:
            hit_light_on_list = curr_light_on_list
            hit_pos = cap.get_position()-1
            hit_pos_frame = frame

        if not curr_light_on:
            return hit_pos, hit_light_on_list, hit_pos_frame

    return hit_pos, hit_light_on_list, hit_pos_frame


def find_hit_position(cap: CV2VideoCapture):
    while cap.get_position() < cap.__len__():
        frame = cap.read()
        light_on, light_on_list = HitChecker.check(frame, including_white=False)
        if light_on:
            return cap.get_position() - 1, light_on_list

    return -1, []


def get_scores(frame):
    left_score = getDigit(frame[309*2:325*2, 265*2:285*2])
    right_score = getDigit(frame[309*2:325*2, 355*2:375*2])
    # print(left_score, right_score)
    return left_score, right_score


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


def find_post_hit_score(cap: CV2VideoCapture):
    prevFrame=None
    while cap.get_position() < cap.__len__():
        frame = cap.read()
        light_on, light_on_list = HitChecker.check(frame, including_white=False)
        if light_on:
            left_score, right_score = get_scores(prevFrame)
            return cap.get_position()-1, left_score, right_score

        prevFrame = frame

    left_score, right_score = get_scores(prevFrame)
    return -1, left_score, right_score


# def get_caption_from_video(cap):
#     orig_pos = cap.get_position()
#     orig_frame = cap.skip(30).read()
#
#     # find next hit
#     hit_end_pos = find_hit_end(cap)
#     cap.set_position(hit_end_pos+1)
#     next_hit_pos = find_hit_position(cap)
#
#     # find score change
#     cap.set_position(orig_pos)
#     prev_l_score, prev_r_score = get_scores(orig_frame)
#     score_changed_pos, l_score, r_score = find_score_change(cap, 15)
#
#     if score_changed_pos == -1 or score_changed_pos > next_hit_pos: # score didn't changed
#         l_score, r_score = prev_l_score, prev_r_score
#
#     if l_score - prev_l_score > 1 or r_score - prev_r_score > 1:
#         raise Exception("Missed score")
#
#     cap.set_position(orig_pos)
#     hit_type = check_lights(orig_frame)
#     if hit_type=="No-No":
#         raise Exception("got No-No")
#
#     if hit_type == "On-On" or hit_type == "On-Off" or hit_type == "Off-On":
#         return caption(hit_type, prev_l_score, prev_r_score, l_score, r_score)
#     else:
#         return None
