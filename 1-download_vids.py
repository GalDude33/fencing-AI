import glob

from pytube import YouTube
import os

## Run this to download all the videos from youtube.

import signal
import time
import traceback


## Timeout for use with try/except so that pytube doesn't randomly freeze.
class Timeout():
    """Timeout class using ALARM signal."""

    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()


# Create all the directories needed if they don't yet exist
# The only non automatic step is right before optical flow production. Some data will be in 'training quarantine', some in 'more_training data'
# (i.e, the extra augmented data). Up to the user to copy these over to final training_clips when ready. 
directories = ['precut', 'videos', 'training_quarantine', 'more_training_data', 'final_training_clips', 'optical_flow',
               'preinception_data', 'final_training_data', 'training_data']
for dirs in directories:
    if not os.path.exists(dirs):
        os.makedirs(dirs)

text_file = open("sabre_videos.txt", "r")
vids = text_file.readlines()
print("First 3 links:", vids[:3])
text_file.close()


# alredy_downloaded = []
# with open('/home/galdude33/console_output_vids', 'r') as f:
#     for line in f:
#         if 'Downloaded' in line:
#             alredy_downloaded.append(line.split()[1])
alredy_downloaded = [_.strip().split('/')[-1].split('.')[0] for _ in glob.glob(os.getcwd() + '/precut/*.mp4')]

# Loop through all the videos, download them and put them in the precut folder.
for i in (_ for _ in vids if _.strip().split('v=')[-1] not in alredy_downloaded):
    try:
        with Timeout(1200):
            start = time.time()
            yt = YouTube(i)
            stream = yt.streams. \
                filter(progressive=True,
                       file_extension='mp4',
                       resolution='720p').first()
            stream.download(os.getcwd() + '/precut/', i.strip().split('v=')[-1])
            print("Downloaded: ", i.strip(), "   ", (time.time() - start), "s")
    except:
        traceback.print_exc()
        print("Failed-", i)
