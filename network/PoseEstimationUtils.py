import numpy as np
import json
import cv2
import math


# Heatmap indices to find each limb (joint connection). Eg: limb_type=1 is
# Neck->LShoulder, so joint_to_limb_heatmap_relationship[1] represents the
# indices of heatmaps to look for joints: neck=1, LShoulder=5
joint_to_limb_heatmap_relationship = [[1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
 [8, 12], [12, 13], [13, 14], [1, 0], [0, 15], [15, 17], [0, 16], [16, 18], [2, 17],
 [5, 18], [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]]

# Color code used to plot different joints and limbs (eg: joint_type=3 and
# limb_type=3 will use colors[3])
colors =    [   [255,     0,    85],
                [255,     0,     0],
                [255,    85,     0],
                [255,   170,     0],
                [255,   255,     0],
                [170,   255,     0],
                [ 85,   255,     0],
                [  0,   255,     0],
                [255,     0,     0],
                [  0,   255,    85],
                [  0,   255,   170],
                [  0,   255,   255],
                [  0,   170,   255],
                [  0,    85,   255],
                [  0,     0,   255],
                [255,     0,   170],
                [170,     0,   255],
                [255,     0,   255],
                [ 85,     0,   255],
                [  0,     0,   255],
                [  0,     0,   255],
                [  0,     0,   255],
                [  0,   255,   255],
                [  0,   255,   255],
                [  0,   255,   255],
                [0, 255, 255]]

NUM_LIMBS = len(joint_to_limb_heatmap_relationship)
NUM_POINTS = 25
IMG_SHAPE=(1280.0, 720.0)


def convert_points_to_lines(coords_point_arr):
    people_num = coords_point_arr.shape[0]
    coords_point_pair_arr = np.zeros((people_num, len(joint_to_limb_heatmap_relationship), 2, 2))

    for i, pair in enumerate(joint_to_limb_heatmap_relationship):
        coords_point_pair_arr[:, i] = np.append(arr=coords_point_arr[:, pair[0], :], values=coords_point_arr[:, pair[1], :], axis=1) \
            .reshape(people_num, 2, 2)

    for p in range(people_num):
        for l in range(len(joint_to_limb_heatmap_relationship)):
            curr_line = coords_point_pair_arr[p, l]
            if (curr_line[0] == np.array([0, 0])).min() or (curr_line[1] == np.array([0, 0])).min():
                coords_point_pair_arr[p, l] = np.array([[0., 0.], [0., 0.]])

    return coords_point_pair_arr


def filterFencingPlayers(coords_point_pair_arr):
    fencing_players_point_pair_coords = coords_point_pair_arr

    if len(coords_point_pair_arr) > 2:
        players_size_mean = np.average(np.sum(np.abs(coords_point_pair_arr[:, :, 0] - coords_point_pair_arr[:, :, 1]), 2), 1)
        fencing_players_ind = np.argsort(players_size_mean)[-3:]
        fencing_players_point_pair_coords = coords_point_pair_arr[fencing_players_ind]

        players_size_mean = players_size_mean[fencing_players_ind]
        players_y_mean = np.mean(np.mean(fencing_players_point_pair_coords[:, :, :, 1], 1), 1)

        ind1_by_size, ind2_by_size = getTwoClosestValuesInArr(players_size_mean)
        ind1_by_y_loc, ind2_by_y_loc = getTwoClosestValuesInArr(players_y_mean)

        #assert(set([ind1_by_size, ind2_by_size]) == set([ind1_by_y_loc, ind2_by_y_loc]))
        #fencing_players_ind = np.argsort(np.array([np.count_nonzero(p_arr == 0.) for p_arr in curr_fencing_players_coords]))[:2]

        fencing_players_ind = np.array([ind1_by_size, ind2_by_size])
        fencing_players_point_pair_coords = fencing_players_point_pair_coords[fencing_players_ind]

    return fencing_players_point_pair_coords


def load_people_point_pose_arr(file_path):

    with (open(file_path, "rb")) as file:
        people_dict = json.load(file)['people']
        people_point_pose_arr = np.array([np.array(p['pose_keypoints_2d']) for p in people_dict])
        people_num = people_point_pose_arr.shape[0]
        people_point_pose_arr = people_point_pose_arr.reshape(people_num, NUM_POINTS, 3)
        people_point_pose_confidence_arr = people_point_pose_arr[:, :, -1]
        people_point_pose_arr = people_point_pose_arr[:, :, :2]

    return people_point_pose_arr, people_point_pose_confidence_arr


def sort_fencing_players(fencing_players_coords):
    # verify left, right side of players
    first_player_x_mean = np.mean(fencing_players_coords[0, :, :, 0])
    second_player_x_mean = np.mean(fencing_players_coords[1, :, :, 0])

    if first_player_x_mean > second_player_x_mean:
        fencing_players_coords = fencing_players_coords[::-1]

    return fencing_players_coords


def normalize_point_pair_pose_arr(fencing_players_coords):
    fencing_players_coords[:, :, :, 0] = fencing_players_coords[:, :, :, 0] / IMG_SHAPE[0]
    fencing_players_coords[:, :, :, 1] = fencing_players_coords[:, :, :, 1] / IMG_SHAPE[1]
    return fencing_players_coords


def plot_from_pose_coords(img_orig, coords_arr, bool_fast_plot=True, plot_ear_to_shoulder=False):
    canvas = img_orig.copy()  # Make a copy so we don't modify the original image

    # to_plot is the location of all joints found overlaid on top of the
    # original image
    to_plot = canvas.copy() if bool_fast_plot else cv2.addWeighted(
        img_orig, 0.3, canvas, 0.7, 0)

    limb_thickness = 4
    # Last 2 limbs connect ears with shoulders and this looks very weird.
    # Disabled by default to be consistent with original rtpose output
    which_limbs_to_plot = NUM_LIMBS# if plot_ear_to_shoulder else NUM_LIMBS - 2
    for limb_type in range(which_limbs_to_plot):
        for person_ind, person_joint_info in enumerate(coords_arr):
            # joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_type]].astype(
            #    int)
            # if -1 in joint_indices:
            # Only draw actual limbs (connected joints), skip if not
            # connected
            # continue
            # joint_coords[:,0] represents Y coords of both joints;
            # joint_coords[:,1], X coords
            joint_coords = coords_arr[person_ind, limb_type]  # joint_list[joint_indices, 0:2]

            for joint in joint_coords:  # Draw circles at every joint
                cv2.circle(canvas, tuple(joint[0:2].astype(
                    int)), 4, (255, 255, 255), thickness=-1)
                # mean along the axis=0 computes meanYcoord and meanXcoord -> Round
            # and make int to avoid errors
            coords_center = tuple(
                np.round(np.mean(joint_coords, 0)).astype(int))
            # joint_coords[0,:] is the coords of joint_src; joint_coords[1,:]
            # is the coords of joint_dst
            limb_dir = joint_coords[0, :] - joint_coords[1, :]
            limb_length = np.linalg.norm(limb_dir)
            # Get the angle of limb_dir in degrees using atan2(limb_dir_x,
            # limb_dir_y)
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))

            # For faster plotting, just plot over canvas instead of constantly
            # copying it
            cur_canvas = canvas if bool_fast_plot else canvas.copy()
            polygon = cv2.ellipse2Poly(
                coords_center, (int(limb_length / 2), limb_thickness), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[limb_type])
            if not bool_fast_plot:
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return to_plot, canvas


def getFencingPlayersPoseArr(descriptor_file):

    people_point_pose_arr, _ = load_people_point_pose_arr(descriptor_file)
    # filter 2 fencing players and sort them
    coords_point_pair_arr = convert_points_to_lines(people_point_pose_arr)
    fencing_players_coords = filterFencingPlayers(coords_point_pair_arr)

    if len(fencing_players_coords) == 2:#TODO: deal also with less than 2 people
        fencing_players_coords = sort_fencing_players(fencing_players_coords)

    # add normalization#TODO- deal better with normaliztion
    fencing_players_coords = normalize_point_pair_pose_arr(fencing_players_coords)

    return fencing_players_coords


def getTwoClosestValuesInArr(arr):
    map = {}

    for i in range(len(arr)):
        for j in range(len(arr)):
            if i<j:
                map[(i,j)] = np.abs(arr[i]-arr[j])

    return min(map, key=map.get)
