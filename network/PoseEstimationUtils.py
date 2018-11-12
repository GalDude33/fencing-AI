import numpy as np


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

        assert(set([ind1_by_size, ind2_by_size]) == set([ind1_by_y_loc, ind2_by_y_loc]))
        #fencing_players_ind = np.argsort(np.array([np.count_nonzero(p_arr == 0.) for p_arr in curr_fencing_players_coords]))[:2]

        fencing_players_ind = np.array([ind1_by_size, ind2_by_size])
        fencing_players_point_pair_coords = fencing_players_point_pair_coords[fencing_players_ind]

    return fencing_players_point_pair_coords


def getTwoClosestValuesInArr(arr):
    map = {}

    for i in range(len(arr)):
        for j in range(len(arr)):
            if i<j:
                map[(i,j)] = np.abs(arr[i]-arr[j])

    return min(map, key=map.get)
