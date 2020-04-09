import numpy as np
import cv2

def draw_3d_poses(obj_box, tf, img, camK):
    lines = [[0, 1], [0, 2], [0, 4], [1, 5], [1, 3], [2, 6], [2, 3], [3, 7],
             [4, 6], [4, 5], [5, 7], [6, 7]]
    direc = [2, 1, 0, 0, 1, 0, 2, 0, 1, 2, 1, 2]
    proj_2d = np.zeros((8, 2), dtype=np.int)
    tf_pts = (np.matmul(tf[:3, :3], obj_box.T) + tf[:3, 3, np.newaxis]).T
    max_z = np.max(tf_pts[:, 2])
    min_z = np.min(tf_pts[:, 2])
    z_diff = max_z - min_z
    z_mean = (max_z + min_z) / 2
    proj_2d[:, 0] = tf_pts[:, 0] / tf_pts[:, 2] * camK[0, 0] + camK[0, 2]
    proj_2d[:, 1] = tf_pts[:, 1] / tf_pts[:, 2] * camK[1, 1] + camK[1, 2]
    for l_id in range(len(lines)):
        line = lines[l_id]
        dr = direc[l_id]
        mean_z_line = (tf_pts[line[0], 2] + tf_pts[line[1], 2]) / 2
        color_amp = (z_mean - mean_z_line) / z_diff * 255
        color = np.zeros((3), dtype=np.uint8)
        color[dr] = min(128 + color_amp, 255)
        if (color[dr] < 10):
            continue
        cv2.line(img, (proj_2d[line[0], 0], proj_2d[line[0], 1]),
                 (proj_2d[line[1], 0], proj_2d[line[1], 1]),
                 (int(color[0]), int(color[1]), int(color[2])), 2)

    pt_colors = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    for pt_id, color in zip([0, 4, 2, 1], pt_colors):  # origin, x,y,z, points
        pt = proj_2d[pt_id]
        cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (color[0], color[1], color[2]), 5)
    return img

def get_3d_box_points(vertices):

    x_min = np.min(vertices[:, 0])
    y_min = np.min(vertices[:, 1])
    z_min = np.min(vertices[:, 2])
    x_max = np.max(vertices[:, 0])
    y_max = np.max(vertices[:, 1])
    z_max = np.max(vertices[:, 2])
    pts = []
    pts.append([x_min, y_min, z_min])  # 0
    pts.append([x_min, y_min, z_max])  # 1
    pts.append([x_min, y_max, z_min])  # 2
    pts.append([x_min, y_max, z_max])  # 3
    pts.append([x_max, y_min, z_min])  # 4
    pts.append([x_max, y_min, z_max])  # 5
    pts.append([x_max, y_max, z_min])  # 6
    pts.append([x_max, y_max, z_max])  # 7
    if (x_max > 1):  # assume, this is mm scale
        return np.array(pts) * 0.001
    else:
        return np.array(pts)
