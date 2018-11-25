import collections
import json
import os
from time import sleep

import numpy as np
from PIL import Image
from cv2 import cv2

from network.PoseEstimationUtils import getFencingPlayersPoseArr


def createLineIterator(P1, P2):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    # define local variables for readability
    # imageH = img.shape[0]
    # imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa + 1, dXa + 1), 2), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[0] = P1
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[1:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[1:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[0] = P1
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[1:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[1:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        itbuffer[0] = P1
        steepSlope = dYa > dXa
        if steepSlope:
            slope = float(dX) / float(dY)
            if negY:
                itbuffer[1:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[1:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[1:, 0] = (slope * (itbuffer[1:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[1:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[1:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[1:, 1] = (slope * (itbuffer[1:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    # colX = itbuffer[:, 0]
    # colY = itbuffer[:, 1]
    # itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    # itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer


Point = collections.namedtuple('Point', ['x', 'y'])
Line = collections.namedtuple('Line', ['start', 'end'])


class LineOpticalFlow:
    def __init__(self, prev_line, next_line):
        # Pad the data with ones, so that our transformation can do translations too
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]
        X = pad(prev_line)
        Y = pad(next_line)

        # Solve the least squares problem X * A = Y
        # to find our transformation matrix A
        A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)

        self._transform = lambda x: unpad(np.dot(pad(x), A))

    def transform(self, x):
        return self._transform(x)


def draw_optical_flow(prev_line, next_line, of):
    lof = LineOpticalFlow(prev_line, next_line)
    pts = createLineIterator(prev_line[0], prev_line[1])
    new_pts = lof.transform(pts)
    dist = new_pts - pts
    of[pts.astype(int)[:, 0], pts.astype(int)[:, 1]] = dist


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


from scipy.ndimage import zoom, grey_dilation, maximum_filter, minimum_filter


def pose2flow(pose1, pose2, img_size_x, img_size_y):
    flow = np.zeros([img_size_x, img_size_y, 2])
    for prev_line, next_line in zip(pose1.astype(int), pose2.astype(int)):
        draw_optical_flow(prev_line, next_line, flow)

    flow = maximum_filter(flow, footprint=np.ones([10, 10, 1])) \
           + minimum_filter(flow, footprint=np.ones([10, 10, 1]))

    flow = flow.transpose([1, 0, 2])[:, :, ::-1]

    return flow


if __name__ == '__main__':
    # prev_line = np.array([(25, 25), (40, 50)])
    # next_line = np.array([(10, 20), (0, 10)])
    #
    # flow = np.zeros([51, 51, 2])
    # draw_optical_flow(prev_line, next_line, flow)
    #
    # flow = np.sign(flow) * maximum_filter(np.abs(flow), footprint=np.ones([3, 3, 1]))
    #
    # Image.fromarray(draw_hsv(flow)).show()
    base_dir = '/home/galdude33/Workspace/Dataset/fencing/FinalPoseEstimationResults/jsons1'
    # with open(os.path.join(base_dir, 'zWDt_NIuovQ-9-R-3107-45_keypoints.json')) as f:
    #     json1 = json.load(f)
    # with open(os.path.join(base_dir, 'zWDt_NIuovQ-9-R-3107-46_keypoints.json')) as f:
    #     json2 = json.load(f)
    # zWDt_NIuovQ-9-R-3107
    pose = getFencingPlayersPoseArr([os.path.join(base_dir, '1RLfRb53jeU-1-R-1057-{:02d}_keypoints.json'.format(i))
                                     for i in range(1, 60 + 1)])
    vid = cv2.VideoCapture('/home/galdude33/Documents/1RLfRb53jeU-1-R-1057.mp4')
    _, last_frame = vid.read()
    # last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    for i in range(59):
        lines1 = pose[i]
        lines2 = pose[i + 1]
        flow1 = pose2flow(lines1[0], lines2[0], 1280, 720)
        flow2 = pose2flow(lines1[1], lines2[1], 1280, 720)
        flow = flow1 + flow2
        # Image.fromarray(draw_hsv(flow.transpose([1,0,2]))).show()

        # cv2.imshow('flow', draw_hsv(flow.transpose([1,0,2])[:,:,::-1]))

        _, frame = vid.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_frame = cv2.resize(draw_hsv(flow), (256, 128))
        cv2.imshow('vid', cv2.resize(np.concatenate([frame, last_frame, flow_frame]), (256 * 2, 128 * 3 * 2)))
        last_frame = frame

        cv2.waitKey()
        # sleep(0.25)
    # cv2.destroyAllWindows()
