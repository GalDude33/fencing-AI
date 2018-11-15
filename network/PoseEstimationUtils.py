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


def find_previous_frame_with_two_players(all_fencing_players_point_pair_coords, seq_ind):
    for j in range(seq_ind, -1, -1):
        if get_players_num(all_fencing_players_point_pair_coords[j])==2:
            return j
    return -1


def get_players_total_y_size(fencing_players_point_pair_coords):
    # total_size = y_max - y_min
    fencing_players_point_pair_coords_copy = fencing_players_point_pair_coords.copy()
    fencing_players_point_pair_coords_copy[fencing_players_point_pair_coords_copy == 0.] = np.inf
    y_min = np.min(np.min(fencing_players_point_pair_coords_copy[:, :, :, 1], 2), 1)
    fencing_players_point_pair_coords_copy[fencing_players_point_pair_coords_copy == np.inf] = -np.inf
    y_max = np.max(np.max(fencing_players_point_pair_coords_copy[:, :, :, 1], 2), 1)
    players_y_size = y_max - y_min
    return players_y_size


def sort_players_by_x_loc(people_x_y):
    return people_x_y[np.argsort(people_x_y[:, 0])]


def map_curr_player_to_prev_player(prev_people_x_y, curr_people_x_y):
    prev_curr_player_map = {}

    for curr_p_ind in range(len(curr_people_x_y)):
        curr_prev_x_y_min_diff = np.inf
        curr_prev_x_y_min_diff_prev_ind = -1

        for prev_p_ind in range(len(prev_people_x_y)):
            x_y_diff = np.sum(np.abs(curr_people_x_y[curr_p_ind]-prev_people_x_y[prev_p_ind]))

            if x_y_diff<curr_prev_x_y_min_diff:
                curr_prev_x_y_min_diff = x_y_diff
                curr_prev_x_y_min_diff_prev_ind = prev_p_ind

        prev_curr_player_map[str(curr_p_ind)] = (curr_prev_x_y_min_diff_prev_ind, curr_prev_x_y_min_diff)

    return prev_curr_player_map


def filter_players_with_same_L_R_direction(prev_curr_player_map):
    res = np.zeros((2))

    for trgSideInd in range(2):
        curr_min_diff = np.inf
        curr_min_ind = -1

        for key in prev_curr_player_map:
            if trgSideInd==prev_curr_player_map[key][0]:
                if prev_curr_player_map[key][1]<curr_min_diff:
                    curr_min_diff = prev_curr_player_map[key][1]
                    curr_min_ind = int(key)

        res[trgSideInd] = curr_min_ind

    return [int(v) for v in res if v>=0]


def filterFencingPlayers(coords_point_pair_lst, people_point_pose_confidence_lst):
    seq_len = len(coords_point_pair_lst)
    all_fencing_players_point_pair_coords = np.zeros((seq_len, 2, NUM_LIMBS, 2, 2))

    for seq_ind in range(seq_len):
        fencing_players_point_pair_coords = coords_point_pair_lst[seq_ind]
        body_confidence = np.mean(people_point_pose_confidence_lst[seq_ind][:, joint_to_limb_heatmap_relationship[0]], 1)
        fencing_players_ind = [i for i in range(len(fencing_players_point_pair_coords))]

        # filter 3 biggest persons by body size
        if len(fencing_players_point_pair_coords) > 3:
            players_body_size = np.sum(np.abs(fencing_players_point_pair_coords[:, 0, 0] - fencing_players_point_pair_coords[:, 0, 1]), 1)
            fencing_players_ind = np.argsort(players_body_size)[-3:]
            fencing_players_ind.sort()
            fencing_players_point_pair_coords = fencing_players_point_pair_coords[fencing_players_ind]

        # filter persons by body confidence
        if len(fencing_players_point_pair_coords)>0:
            candidates_body_confidence = body_confidence[fencing_players_ind]
            fencing_players_ind = np.where(candidates_body_confidence > 0.20)[0]
            fencing_players_ind.sort()
            fencing_players_point_pair_coords = fencing_players_point_pair_coords[fencing_players_ind]

        # filter persons by total size: total_size = y_max - y_min
        if len(fencing_players_point_pair_coords) > 0:
            players_y_size = get_players_total_y_size(fencing_players_point_pair_coords)
            fencing_players_ind = np.where(players_y_size > 105.0)[0]
            fencing_players_ind.sort()
            fencing_players_point_pair_coords = fencing_players_point_pair_coords[fencing_players_ind]

        # filter persons by (x,y) location and players y total size in previous frame with two players
        prev_frame_with_two_players_ind = find_previous_frame_with_two_players(all_fencing_players_point_pair_coords, seq_ind-1)

        if len(fencing_players_point_pair_coords)>0 and prev_frame_with_two_players_ind>-1:
            prev_people_x_y_mean = np.mean(all_fencing_players_point_pair_coords[prev_frame_with_two_players_ind, :, 0, :, :], 1)
            curr_people_x_y_mean = np.mean(fencing_players_point_pair_coords[ :, 0, :, :], 1)

            prev_people_x_y_mean = sort_players_by_x_loc(prev_people_x_y_mean)
            prev_curr_player_map = map_curr_player_to_prev_player(prev_people_x_y_mean, curr_people_x_y_mean)
            fencing_players_ind = filter_players_with_same_L_R_direction(prev_curr_player_map)
            fencing_players_ind.sort()
            fencing_players_point_pair_coords = fencing_players_point_pair_coords[fencing_players_ind]
            curr_people_x_y_mean = curr_people_x_y_mean[fencing_players_ind]

            prev_people_total_y_size = get_players_total_y_size(all_fencing_players_point_pair_coords[prev_frame_with_two_players_ind])
            curr_people_total_y_size = get_players_total_y_size(fencing_players_point_pair_coords)

            fencing_players_ind = []
            for p in range(len(fencing_players_point_pair_coords)):
                curr_x_y = curr_people_x_y_mean[p]
                curr_total_y_size = curr_people_total_y_size[p]

                if len([prev_x_y for prev_x_y in prev_people_x_y_mean if np.sum(np.abs(curr_x_y-prev_x_y))<200.0])>0:
                    if len([prev_total_y_size for prev_total_y_size in prev_people_total_y_size if np.abs(curr_total_y_size - prev_total_y_size) < 70.0])>0:
                        fencing_players_ind.append(p)

            fencing_players_ind.sort()
            fencing_players_point_pair_coords = fencing_players_point_pair_coords[fencing_players_ind]

        # take the two closetst players by y body location
        if len(fencing_players_point_pair_coords) > 2:
            players_y_mean = np.mean(fencing_players_point_pair_coords[:, 0, :, 1], 1)
            ind1_by_y_loc, ind2_by_y_loc = getTwoClosestValuesInArr(players_y_mean)
            fencing_players_ind = np.array([ind1_by_y_loc, ind2_by_y_loc])
            fencing_players_ind.sort()
            fencing_players_point_pair_coords = fencing_players_point_pair_coords[fencing_players_ind]

        for p in range(len(fencing_players_point_pair_coords)):
            all_fencing_players_point_pair_coords[seq_ind, p] = fencing_players_point_pair_coords[p]

    return all_fencing_players_point_pair_coords


def get_frames_to_repair(fencing_players_coords, player_ind):
    max_delta_y = 125.0
    y_mean_player = np.median(np.mean(fencing_players_coords[:, player_ind, 0, :, 1], 1))
    frames_to_repair_for_player = []
    seq_len = fencing_players_coords.shape[0]

    for seq_ind in range(seq_len):
        curr_frame_y_mean_player = np.mean(fencing_players_coords[seq_ind, player_ind, 0, :, 1])

        if np.abs(curr_frame_y_mean_player - y_mean_player) > max_delta_y:
            frames_to_repair_for_player.append(seq_ind)

    return frames_to_repair_for_player


def change_player_problematic_frames(fencing_players_coords, frames_to_repair_for_player, player_ind):
    #seq_len = fencing_players_coords.shape[0]
    if any(frames_to_repair_for_player):
        frames_to_repair_for_player_copy = frames_to_repair_for_player.copy()

        for seq_ind in frames_to_repair_for_player:
            start_frame_to_count = -1#, end_frame_to_count = -1, -1

            for j in range(seq_ind-1, -1, -1):
                if j not in frames_to_repair_for_player_copy:
                    start_frame_to_count = j
                    break

            # for k in range(seq_ind+1, seq_len, 1):
            #     if k not in frames_to_repair_for_player_copy:
            #         end_frame_to_count = k
            #         break

            if start_frame_to_count != -1:# and end_frame_to_count != -1:
                #gap = end_frame_to_count-start_frame_to_count
                #gap1 = seq_ind-start_frame_to_count
                fencing_players_coords[seq_ind, player_ind] = fencing_players_coords[start_frame_to_count, player_ind]#+ \
                #                                      gap1*((fencing_players_coords[end_frame_to_count, player_ind]-fencing_players_coords[start_frame_to_count, player_ind])/gap)

                #frames_to_repair_for_player_copy.remove(seq_ind)

    return fencing_players_coords


def repairFencingPlayersPoses(fencing_players_coords):
    _fencing_players_coords = fencing_players_coords

    frames_to_repair_for_player0 = get_frames_to_repair(_fencing_players_coords, 0)
    frames_to_repair_for_player1 = get_frames_to_repair(_fencing_players_coords, 1)

    if any(frames_to_repair_for_player0):
        for seq_ind in frames_to_repair_for_player0:
            _fencing_players_coords[seq_ind, 0] = np.zeros((NUM_LIMBS, 2, 2))

    if any(frames_to_repair_for_player1):
        for seq_ind in frames_to_repair_for_player1:
            _fencing_players_coords[seq_ind, 1] = np.zeros((NUM_LIMBS, 2, 2))

    _fencing_players_coords = sort_fencing_players(_fencing_players_coords)

    frames_to_repair_for_player0 = get_frames_to_repair(_fencing_players_coords, 0)
    frames_to_repair_for_player1 = get_frames_to_repair(_fencing_players_coords, 1)

    _fencing_players_coords = change_player_problematic_frames(_fencing_players_coords, frames_to_repair_for_player0, 0)
    _fencing_players_coords = change_player_problematic_frames(_fencing_players_coords, frames_to_repair_for_player1, 1)

    return _fencing_players_coords


def load_people_point_pose_arr(file_path):

    with (open(file_path, "rb")) as file:
        people_dict = json.load(file)['people']
        people_point_pose_arr = np.array([np.array(p['pose_keypoints_2d']) for p in people_dict])
        people_num = people_point_pose_arr.shape[0]
        people_point_pose_arr = people_point_pose_arr.reshape(people_num, NUM_POINTS, 3)
        people_point_pose_confidence_arr = people_point_pose_arr[:, :, -1]
        people_point_pose_arr = people_point_pose_arr[:, :, :2]

    return people_point_pose_arr, people_point_pose_confidence_arr


def get_players_num(fencing_players_coords):
    playersNum = 0

    for p in range(len(fencing_players_coords)):
        curr_player_candidate = fencing_players_coords[p]
        if np.max(curr_player_candidate)>0:
            playersNum+=1

    return playersNum


def get_player_ind(fencing_players_coords):
    playerIndices = []

    for p in range(len(fencing_players_coords)):
        curr_player_candidate = fencing_players_coords[p]
        if np.max(curr_player_candidate) > 0:
            playerIndices.append(p)

    return playerIndices


def sort_fencing_players(fencing_players_coords):
    # verify left, right side of players
    seq_len = fencing_players_coords.shape[0]

    for seq_ind in range(seq_len):
        curr_fencing_players_coords = fencing_players_coords[seq_ind]
        _fencing_players_coords = np.zeros((2, NUM_LIMBS, 2, 2))

        if get_players_num(curr_fencing_players_coords) == 0:
            fencing_players_coords[seq_ind] = _fencing_players_coords
            continue

        if get_players_num(curr_fencing_players_coords) == 1:
            old_player_ind = get_player_ind(curr_fencing_players_coords)[0]
            player_x_mean = np.mean(curr_fencing_players_coords[old_player_ind, 0, :, 0])

            prev_frame_player0_x_mean = 0
            prev_frame_player1_x_mean = 0
            j = seq_ind - 1
            while j>=0 and (prev_frame_player0_x_mean==0 or prev_frame_player1_x_mean==0):
                prev_frame_player0_x_mean = np.mean(fencing_players_coords[j, 0, 0, :, 0])
                prev_frame_player1_x_mean = np.mean(fencing_players_coords[j, 1, 0, :, 0])
                j=j-1

            if prev_frame_player0_x_mean>0 and prev_frame_player1_x_mean>0:
                curr_player_ind = np.argmin(np.array([np.abs(player_x_mean-prev_frame_player0_x_mean), np.abs(player_x_mean-prev_frame_player1_x_mean)]))
            else:
                curr_player_ind = 0 if player_x_mean<IMG_SHAPE[0]/2 else 1

            _fencing_players_coords[curr_player_ind] = curr_fencing_players_coords[old_player_ind]
            fencing_players_coords[seq_ind] = _fencing_players_coords
            continue

        #recognized two players
        first_player_x_mean = np.mean(curr_fencing_players_coords[0, 0, :, 0])
        second_player_x_mean = np.mean(curr_fencing_players_coords[1, 0, :, 0])

        if first_player_x_mean > second_player_x_mean:
            curr_fencing_players_coords = curr_fencing_players_coords[::-1]

        fencing_players_coords[seq_ind] = curr_fencing_players_coords

    return fencing_players_coords


def normalize_point_pair_pose_arr(fencing_players_coords):
    fencing_players_coords[:, :, :, :, 0] = fencing_players_coords[:, :, :, :, 0] / IMG_SHAPE[0]
    fencing_players_coords[:, :, :, :, 1] = fencing_players_coords[:, :, :, :, 1] / IMG_SHAPE[1]
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


def getFencingPlayersPoseArr(descriptor_files):
    # filter 2 fencing players and sort them
    descriptor_files.sort()

    seq_len = len(descriptor_files)
    all_coords_point_pair_lst = []
    all_people_point_pose_confidence_lst = []

    for seq_ind in range(seq_len):
        people_point_pose_arr, people_point_pose_confidence_arr = load_people_point_pose_arr(descriptor_files[seq_ind])
        coords_point_pair_arr = convert_points_to_lines(people_point_pose_arr).astype(np.float32)

        all_coords_point_pair_lst.append(coords_point_pair_arr)
        all_people_point_pose_confidence_lst.append(people_point_pose_confidence_arr)

    fencing_players_coords = filterFencingPlayers(all_coords_point_pair_lst, all_people_point_pose_confidence_lst)

    fencing_players_coords = sort_fencing_players(fencing_players_coords)
    fencing_players_coords = repairFencingPlayersPoses(fencing_players_coords)

    return fencing_players_coords


def getTwoClosestValuesInArr(arr):
    map = {}

    for i in range(len(arr)):
        for j in range(len(arr)):
            if i<j:
                map[(i,j)] = np.abs(arr[i]-arr[j])

    return min(map, key=map.get)
