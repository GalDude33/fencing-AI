# Cuts the videos into a set of short clips where each actual hit happens. These clips are used by the data_labeller to
# label the clips where the referee had to distinguish whos priority.
import glob
import subprocess as sp
import os
import cv2
from pylab import *
from pathlib import Path
from DigitRecognizer import getDigit
from PIL import Image


green_box = cv2.imread("greenbox.png")
red_box = cv2.imread("redbox.png")
white_box = cv2.imread("whitebox.png")

# Prelim info, FOTR light box is frame[329:334, 380:500]
# therefore FOTL box is frame[329:334, 140:260]
# FOTL OFF-TARGET frame[337:348, 234:250]
# FOTR OFF-TARGET frame[337:348, 390:406]
#      scoreleft = frame[310:325, 265:285]
#      scoreRight = frame[310:325, 355:375]

FFMPEG_BIN = "ffmpeg"

fps = str(13)

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

already_processed = 0
for vid in glob.glob(os.getcwd() + "/precut/" + "*.mp4"):
    if int(Path(vid).stem) >= already_processed:
        print("Video:", vid)
        clips_recorded = 0
        recording_mode = False

        cap = cv2.VideoCapture(str(vid))

        cap_end_point = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Length of Vid:", cap_end_point, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
        cap_end_point = cap_end_point - jump_length  # ensures videos don't overrun
        print("Beginning to cut...")

        position = 0

        while position < cap_end_point:
            cap = cv2.VideoCapture(str(vid))

            print(position, "big while loop", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            if position == cap_end_point:
                print("should not be here")
                break

            if recording_mode:
                print("Recording Mode On")
                output_file = 'videos/' + Path(vid).stem + "-" + str(clips_recorded) + '.mp4'

                command = [FFMPEG_BIN,
                           '-y',
                           '-f', 'rawvideo',
                           '-vcodec', 'rawvideo',
                           '-s', '640*360',
                           '-pix_fmt', 'bgr24',
                           '-r', fps,
                           '-i', '-',
                           '-an',
                           '-vcodec', 'mpeg4',
                           '-b:v', '5000k',
                           output_file]

                frames_till_video_end = jump_length

                proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                cap.set(cv2.CAP_PROP_FPS, 10000)

                while cap.isOpened():
                    ret, frame = cap.read()
                    position = position + 1

                    if not recording_mode:
                        if position % 100 == 0:
                            print(position)
                            print(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        # this check is here because the vid should be prevented from 
                        # starting a clip less than frames_till_vid_end away from 
                        # the end
                        if position == cap_end_point:
                            break
                        elif cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap_end_point:
                            print("break")
                            position = cap.get(cv2.CAP_PROP_POS_FRAMES)
                            break
                        try:
                            if (np.sum(abs(frame[337:348, 234:250].astype(int) - white_box.astype(int))) <= 7000) or (
                                    np.sum(abs(
                                        frame[337:348, 390:406].astype(int) - white_box.astype(int))) <= 7000) or (
                                    np.sum(abs(
                                        frame[330:334, 380:500].astype(int) - green_box.astype(int))) <= 40000) or (
                                    np.sum(abs(frame[330:334, 140:260].astype(int) - red_box.astype(int))) <= 40000):

                                left_score = int(getDigit(frame[309:325, 265:285]))
                                right_score = int(getDigit(frame[309:325, 355:375]))
                                print(left_score, right_score)

                                if (left_score == 15) or (right_score == 15):
                                    print("don't record this hit")
                                    position = position + 25
                                    break
                                elif (left_score == 0) and (right_score == 0):
                                    print("don't record this hit")
                                    position = position + 25
                                    break
                                else:
                                    print("recording hit")

                                    # jump back 50 frames to the action of the hit
                                    position = position - 50

                                    print("Light seen, position-", position)
                                    recording_mode = True
                                    break
                        except:
                            break

                    elif recording_mode:
                        if frames_till_video_end >= hide_length:
                            if position % 2 == 0:
                                proc.stdin.write(frame.tostring())

                        frames_till_video_end = frames_till_video_end - 1
                        if frames_till_video_end == 0:
                            print("finished clip")
                            recording_mode = False
                            proc.stdin.close()
                            proc.stderr.close()
                            print(clips_recorded)
                            clips_recorded = clips_recorded + 1
                            break
            else:
                print("Failed to open video")

            cap.release()
