import numpy as np
from typing import List, Optional, Tuple
from .constants import N_UPPER_BODY_POSE_LANDMARKS


def calculate_keypoints_center(points: np.ndarray, num_pose_landmarks_for_center: int) -> Tuple[float, float, bool]:
    pose_points = points[0:N_UPPER_BODY_POSE_LANDMARKS]
    hand_points_start_index = N_UPPER_BODY_POSE_LANDMARKS
    other_points_for_center = points[hand_points_start_index:]
    points_for_center_pose_part = pose_points[0:num_pose_landmarks_for_center]
    points_to_calculate_center_list = [points_for_center_pose_part]
    if other_points_for_center.shape[0] > 0:
        points_to_calculate_center_list.append(other_points_for_center)
    
    center_x, center_y = 0.0, 0.0
    can_calculate_center = False

    if points_to_calculate_center_list:
        points_to_calculate_center_concat = np.concatenate(points_to_calculate_center_list, axis=0)
        valid_center_points_mask = np.any(points_to_calculate_center_concat != 0, axis=1)
        valid_center_points = points_to_calculate_center_concat[valid_center_points_mask]

        if valid_center_points.shape[0] > 0:
            center_x = np.median(valid_center_points[:, 0])
            center_y = np.median(valid_center_points[:, 1])
            can_calculate_center = True
        else:
            all_valid_points_mask = np.any(points != 0, axis=1)
            all_valid_points = points[all_valid_points_mask]
            if all_valid_points.shape[0] > 0:
                center_x = np.median(all_valid_points[:, 0])
                center_y = np.median(all_valid_points[:, 1])
                can_calculate_center = True
    return center_x, center_y, can_calculate_center