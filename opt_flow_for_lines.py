import gzip
import os
import pickle
from multiprocessing.dummy import Pool

import cv2
from glob2 import glob
from pathlib2 import Path
from tqdm import tqdm
import numpy as np


class VideoCaptureFromPhotos:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.curr_i = -1

    def read(self):
        try:
            self.curr_i += 1
            img = cv2.imread(os.path.join(self.folder_path, str(self.curr_i)+'.png'))
            return img
        except FileNotFoundError:
            return None


def optical_flow(cap, output_folder):
    frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cap.read()
    ind = 0

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    # out = cv2.VideoWriter(os.path.join(output_folder, 'opt_flow.avi'),
    #                       fourcc, 20, (frame1.shape[0], frame1.shape[1]))

    while frame2 is not None:
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow[prvs < 15] = 0

        #new_size = flow.shape[:-1]+(3,)
        #flow.resize(new_size, refcheck=False)
        # cv2.imwrite(os.path.join(output_folder, 'opt_flow_{}.png'.format(str(ind))),
        #             flow)
        #out.write(np.uint8(flow))
        with gzip.open(os.path.join(output_folder, 'opt_flow_{}.pkl.gz'.format(str(ind))), 'wb') as f:
            pickle.dump(flow, f)

        prvs = next
        frame2 = cap.read()
        ind += 1


def optical_flow_from_folder(folder_path):
    cap = VideoCaptureFromPhotos(folder_path)
    optical_flow(cap, folder_path)


def load_gzip_pickle(p):
    with gzip.open(p, 'rb') as f:
        return pickle.load(f)

def save_gzip_pickle(o, p):
    with gzip.open(p, 'wb') as f:
        return pickle.dump(o, f)

def merge_opt_flow_files(folder_path):
    files = sorted(glob(os.path.join(folder_path, 'opt_flow_*.pkl.gz')),
           key=lambda p: int(Path(p).name.split('.')[-3].split('_')[-1]))

    opts = [load_gzip_pickle(_) for _ in files]
    opts = [np.expand_dims(_, axis=0) for _ in opts]
    merged_opts = np.concatenate(opts, axis=0)

    save_gzip_pickle(merged_opts, os.path.join(folder_path, 'opt_flows.pkl.gz'))


def merge_photos(folder_path):
    files = sorted(glob(os.path.join(folder_path, '*.png')),
                   key=lambda p: int(Path(p).name.split('.')[0]))
    frames = [np.expand_dims(_, axis=0) for _ in (cv2.imread(_) for _ in files)]
    merged_opts = np.concatenate(frames, axis=0)

    save_gzip_pickle(merged_opts, os.path.join(folder_path, 'frames.pkl.gz'))


if __name__ == '__main__':
    max_parallel = 8
    pool = Pool(max_parallel)
    db_path = '../Dataset/poses_clips_reduced_players_different_channel'
    clip_paths = list(glob(os.path.join(db_path, '*', '*', '*')))
    with tqdm(total=len(clip_paths)) as pbar:
        for i, _ in enumerate(pool.imap_unordered(merge_photos, clip_paths)):
            pbar.update()
