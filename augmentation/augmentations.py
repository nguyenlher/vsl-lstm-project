import numpy as np
import random
import math
import logging
from typing import List, Optional, Tuple
from .constants import N_TOTAL_LANDMARKS, N_UPPER_BODY_POSE_LANDMARKS, POSE_LM_LEFT_SHOULDER, POSE_LM_RIGHT_SHOULDER, POSE_LM_LEFT_ELBOW, POSE_LM_RIGHT_ELBOW, POSE_LM_LEFT_WRIST, POSE_LM_RIGHT_WRIST, N_HAND_LANDMARKS
from .utils import calculate_keypoints_center
from .ik_solver import solve_2d_arm_ik

def augment_keypoints_scaling(
    keypoints_sequence: List[Optional[np.ndarray]],
    scale_factor_range: Tuple[float, float] = (0.7, 1.26),
    num_total_landmarks: int = N_TOTAL_LANDMARKS,
    num_pose_landmarks_for_center: int = N_UPPER_BODY_POSE_LANDMARKS - 2,
    normalize_to_01: bool = True
) -> List[Optional[np.ndarray]]:

    processed_sequence = []
    if not keypoints_sequence:
        return processed_sequence

    current_scale_factor = random.uniform(scale_factor_range[0], scale_factor_range[1])

    if current_scale_factor <= 0:
        logging.warning("Scale factor must be positive. Returning original sequence.")
        if normalize_to_01:
             temp_sequence = []
             for frame_flat in keypoints_sequence:
                if frame_flat is None:
                    temp_sequence.append(None)
                    continue
                try:
                    points_to_norm = frame_flat.copy().reshape(num_total_landmarks, 3)
                    valid_xy_mask_norm = np.any(points_to_norm[:, :2] != 0, axis=1)
                    if np.any(valid_xy_mask_norm):
                        x_coords = points_to_norm[valid_xy_mask_norm, 0]
                        y_coords = points_to_norm[valid_xy_mask_norm, 1]

                        min_x, max_x = np.min(x_coords), np.max(x_coords)
                        min_y, max_y = np.min(y_coords), np.max(y_coords)

                        if (max_x - min_x) > 1e-7:
                            points_to_norm[valid_xy_mask_norm, 0] = (x_coords - min_x) / (max_x - min_x)
                        elif x_coords.size > 0:
                            points_to_norm[valid_xy_mask_norm, 0] = 0.5

                        if (max_y - min_y) > 1e-7:
                            points_to_norm[valid_xy_mask_norm, 1] = (y_coords - min_y) / (max_y - min_y)
                        elif y_coords.size > 0:
                            points_to_norm[valid_xy_mask_norm, 1] = 0.5
                    temp_sequence.append(points_to_norm.flatten())
                except Exception:
                    temp_sequence.append(frame_flat.copy())
             return temp_sequence
        return [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in keypoints_sequence]

    for frame_keypoints_flat in keypoints_sequence:
        if frame_keypoints_flat is None:
            processed_sequence.append(None)
            continue

        if not isinstance(frame_keypoints_flat, np.ndarray) or \
           frame_keypoints_flat.shape != (num_total_landmarks * 3,):
            processed_sequence.append(frame_keypoints_flat.copy())
            continue

        try:
            current_points_3d = frame_keypoints_flat.copy().reshape(num_total_landmarks, 3)
        except ValueError:
            processed_sequence.append(frame_keypoints_flat.copy())
            continue

        center_x, center_y, can_calculate_center = calculate_keypoints_center(current_points_3d, num_pose_landmarks_for_center)

        if can_calculate_center:
            all_valid_points_mask_for_scaling = np.any(current_points_3d != 0, axis=1)
            
            if np.any(all_valid_points_mask_for_scaling):
                x_all_valid = current_points_3d[all_valid_points_mask_for_scaling, 0]
                y_all_valid = current_points_3d[all_valid_points_mask_for_scaling, 1]


                x_trans = x_all_valid - center_x
                y_trans = y_all_valid - center_y
                x_scaled = x_trans * current_scale_factor
                y_scaled = y_trans * current_scale_factor
                new_x_all_valid = x_scaled + center_x
                new_y_all_valid = y_scaled + center_y

                current_points_3d[all_valid_points_mask_for_scaling, 0] = new_x_all_valid
                current_points_3d[all_valid_points_mask_for_scaling, 1] = new_y_all_valid

        if normalize_to_01:
            valid_xy_mask_norm = np.any(current_points_3d[:, :2] != 0, axis=1)

            if np.any(valid_xy_mask_norm):
                x_coords = current_points_3d[valid_xy_mask_norm, 0]
                y_coords = current_points_3d[valid_xy_mask_norm, 1]

                min_x, max_x = np.min(x_coords), np.max(x_coords)
                min_y, max_y = np.min(y_coords), np.max(y_coords)


                if (max_x - min_x) > 1e-7:
                    current_points_3d[valid_xy_mask_norm, 0] = (x_coords - min_x) / (max_x - min_x)
                elif x_coords.size > 0:
                    current_points_3d[valid_xy_mask_norm, 0] = 0.5


                if (max_y - min_y) > 1e-7:
                    current_points_3d[valid_xy_mask_norm, 1] = (y_coords - min_y) / (max_y - min_y)
                elif y_coords.size > 0:
                    current_points_3d[valid_xy_mask_norm, 1] = 0.5


        processed_frame_flat_output = current_points_3d.flatten()

        if np.isnan(processed_frame_flat_output).any() or np.isinf(processed_frame_flat_output).any():
            processed_sequence.append(frame_keypoints_flat.copy())
        else:
            processed_sequence.append(processed_frame_flat_output)

    return processed_sequence


def augment_keypoints_rotation(
    keypoints_sequence: List[Optional[np.ndarray]],
    angle_degrees_range: Tuple[float, float] = (-15.0, 15.0),
    num_total_landmarks: int = N_TOTAL_LANDMARKS,
    num_pose_landmarks_for_center: int = N_UPPER_BODY_POSE_LANDMARKS - 2
) -> List[Optional[np.ndarray]]:

    rotated_sequence = []
    if not keypoints_sequence:
        return rotated_sequence

    angle_deg = random.uniform(angle_degrees_range[0], angle_degrees_range[1])

    angle_rad = math.radians(angle_deg)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    for frame_keypoints_flat in keypoints_sequence:
        if frame_keypoints_flat is None:
            rotated_sequence.append(None)
            continue

        if not isinstance(frame_keypoints_flat, np.ndarray) or \
           frame_keypoints_flat.shape != (num_total_landmarks * 3,):
            rotated_sequence.append(frame_keypoints_flat.copy())
            continue

        try:
            all_points = frame_keypoints_flat.copy().reshape(num_total_landmarks, 3)
        except ValueError:
            rotated_sequence.append(frame_keypoints_flat.copy())
            continue


        center_x, center_y, can_calculate_center = calculate_keypoints_center(all_points, num_pose_landmarks_for_center)

        if not can_calculate_center:
            rotated_sequence.append(frame_keypoints_flat.copy())


        rotated_all_points = all_points.copy()
        all_valid_points_mask_for_rotation = np.any(all_points != 0, axis=1)


        x_original_valid = all_points[all_valid_points_mask_for_rotation, 0]
        y_original_valid = all_points[all_valid_points_mask_for_rotation, 1]


        x_translated = x_original_valid - center_x
        y_translated = y_original_valid - center_y


        x_rotated = x_translated * cos_angle - y_translated * sin_angle
        y_rotated = x_translated * sin_angle + y_translated * cos_angle


        new_x_all_valid = x_rotated + center_x
        new_y_all_valid = y_rotated + center_y


        rotated_all_points[all_valid_points_mask_for_rotation, 0] = new_x_all_valid
        rotated_all_points[all_valid_points_mask_for_rotation, 1] = new_y_all_valid


        rotated_frame_flat_output = rotated_all_points.flatten()

        if np.isnan(rotated_frame_flat_output).any() or np.isinf(rotated_frame_flat_output).any():
            rotated_sequence.append(frame_keypoints_flat.copy())
        else:
            rotated_sequence.append(rotated_frame_flat_output)

    return rotated_sequence


def augment_keypoints_translation(
    keypoints_sequence: List[Optional[np.ndarray]],
    translate_x_range: Tuple[float, float] = (-0.05, 0.05),
    translate_y_range: Tuple[float, float] = (-0.05, 0.05),
    clip_to_01: bool = True,
    num_total_landmarks: int = N_TOTAL_LANDMARKS,
) -> List[Optional[np.ndarray]]:
    translated_sequence = []
    if not keypoints_sequence:
        return translated_sequence

    dx = random.uniform(translate_x_range[0], translate_x_range[1])
    dy = random.uniform(translate_y_range[0], translate_y_range[1])

    for frame_keypoints_flat in keypoints_sequence:
        if frame_keypoints_flat is None:
            translated_sequence.append(None)
            continue

        if not isinstance(frame_keypoints_flat, np.ndarray) or \
           frame_keypoints_flat.shape != (num_total_landmarks * 3,):
            translated_sequence.append(frame_keypoints_flat.copy())
            continue

        try:
            all_points = frame_keypoints_flat.copy().reshape(num_total_landmarks, 3)
        except ValueError:
            translated_sequence.append(frame_keypoints_flat.copy())
            continue

        translated_all_points = all_points.copy()

        valid_points_mask = np.any(all_points != 0, axis=1)


        translated_all_points[valid_points_mask, 0] += dx
        translated_all_points[valid_points_mask, 1] += dy

        if clip_to_01:
            translated_all_points[valid_points_mask, 0] = np.clip(translated_all_points[valid_points_mask, 0], 0.0, 1.0)
            translated_all_points[valid_points_mask, 1] = np.clip(translated_all_points[valid_points_mask, 1], 0.0, 1.0)

        translated_frame_flat_output = translated_all_points.flatten()

        if np.isnan(translated_frame_flat_output).any() or np.isinf(translated_frame_flat_output).any():
            translated_sequence.append(frame_keypoints_flat.copy())
        else:
            translated_sequence.append(translated_frame_flat_output)

    return translated_sequence


def augment_keypoints_time_stretch(
    keypoints_sequence: List[Optional[np.ndarray]],
    speed_factor_range: Tuple[float, float] = (0.8, 1.2),
) -> List[Optional[np.ndarray]]:
    perturbed_sequence = []
    if not keypoints_sequence or all(kp is None for kp in keypoints_sequence):
        return keypoints_sequence

    valid_frames = [kp for kp in keypoints_sequence if kp is not None]
        return keypoints_sequence

    original_num_valid_frames = len(valid_frames)

    current_speed_factor = random.uniform(speed_factor_range[0], speed_factor_range[1])

    if current_speed_factor <= 0:
        logging.warning("Speed factor must be positive. Returning original sequence.")
        return [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in keypoints_sequence]
        return [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in keypoints_sequence]


    num_new_frames = int(round(original_num_valid_frames / current_speed_factor))

        if original_num_valid_frames > 0:
            perturbed_sequence.append(valid_frames[0].copy() if valid_frames[0] is not None else None)
        return perturbed_sequence

    original_indices = np.linspace(0, original_num_valid_frames - 1, num_new_frames)
    resampled_indices = np.round(original_indices).astype(int)
    resampled_indices = np.clip(resampled_indices, 0, original_num_valid_frames - 1)

    for res_idx in resampled_indices:
        perturbed_sequence.append(valid_frames[res_idx].copy())

    return perturbed_sequence


def augment_inter_hand_distance(
    keypoints_sequence: List[Optional[np.ndarray]],
    total_dx_change_range: Tuple[float, float] = (-0.1, 0.1),
    overall_dy_shift_range: Tuple[float, float] = (-0.03, 0.03),
    clip_to_01: bool = True,
    num_total_landmarks: int = N_TOTAL_LANDMARKS,
) -> List[Optional[np.ndarray]]:
    augmented_sequence = []
    if not keypoints_sequence:
        return augmented_sequence

    current_total_dx_change = random.uniform(total_dx_change_range[0], total_dx_change_range[1])

    current_overall_dy_shift = random.uniform(overall_dy_shift_range[0], overall_dy_shift_range[1])

    for frame_keypoints_flat in keypoints_sequence:
        if frame_keypoints_flat is None:
            augmented_sequence.append(None)
            continue
        if not isinstance(frame_keypoints_flat, np.ndarray) or \
           frame_keypoints_flat.shape != (num_total_landmarks * 3,):
            augmented_sequence.append(frame_keypoints_flat.copy())
            continue
        try:
            all_points_orig = frame_keypoints_flat.copy().reshape(num_total_landmarks, 3)
        except ValueError:
            augmented_sequence.append(frame_keypoints_flat.copy())
            continue

        augmented_points = all_points_orig.copy()

        s_l_orig_xy = all_points_orig[POSE_LM_LEFT_SHOULDER, 0:2].copy()
        e_l_orig_xy = all_points_orig[POSE_LM_LEFT_ELBOW, 0:2].copy()
        w_l_orig_xy = all_points_orig[POSE_LM_LEFT_WRIST, 0:2].copy()

        s_r_orig_xy = all_points_orig[POSE_LM_RIGHT_SHOULDER, 0:2].copy()
        e_r_orig_xy = all_points_orig[POSE_LM_RIGHT_ELBOW, 0:2].copy()
        w_r_orig_xy = all_points_orig[POSE_LM_RIGHT_WRIST, 0:2].copy()

        left_arm_key_points_valid = np.all(s_l_orig_xy != 0) and np.all(e_l_orig_xy != 0) and np.all(w_l_orig_xy != 0)
        right_arm_key_points_valid = np.all(s_r_orig_xy != 0) and np.all(e_r_orig_xy != 0) and np.all(w_r_orig_xy != 0)


        if np.all(w_l_orig_xy != 0) and np.all(w_r_orig_xy != 0):
            current_mid_wrists_x = (w_l_orig_xy[0] + w_r_orig_xy[0]) / 2
            x_left = min(w_l_orig_xy[0], w_r_orig_xy[0])
            x_right = max(w_l_orig_xy[0], w_r_orig_xy[0])
            current_dist_wrists_x = x_right - x_left

            target_dist_wrists_x = current_dist_wrists_x + current_total_dx_change
            if target_dist_wrists_x < 0.01:
                target_dist_wrists_x = 0.01

            if w_l_orig_xy[0] <= w_r_orig_xy[0]:
                w_l_target_x = current_mid_wrists_x - target_dist_wrists_x / 2
                w_r_target_x = current_mid_wrists_x + target_dist_wrists_x / 2
            else:
                w_r_target_x = current_mid_wrists_x - target_dist_wrists_x / 2
                w_l_target_x = current_mid_wrists_x + target_dist_wrists_x / 2
        else:
            w_l_target_x = w_l_orig_xy[0]
            w_r_target_x = w_r_orig_xy[0]


        if left_arm_key_points_valid:
            len_l_upper = np.linalg.norm(e_l_orig_xy - s_l_orig_xy)
            len_l_forearm = np.linalg.norm(w_l_orig_xy - e_l_orig_xy)
            w_l_target_xy_for_ik = np.array([w_l_target_x, w_l_orig_xy[1]])

            e_l_new_xy = solve_2d_arm_ik(s_l_orig_xy, w_l_target_xy_for_ik, len_l_upper, len_l_forearm, e_l_orig_xy, w_l_orig_xy)

            if e_l_new_xy is not None:
                dx_wrist_l = w_l_target_xy_for_ik[0] - w_l_orig_xy[0]
                dy_wrist_l = w_l_target_xy_for_ik[1] - w_l_orig_xy[1]

                augmented_points[POSE_LM_LEFT_ELBOW, 0:2] = e_l_new_xy
                augmented_points[POSE_LM_LEFT_WRIST, 0:2] = w_l_target_xy_for_ik

                idx_lh_start = N_UPPER_BODY_POSE_LANDMARKS
                idx_lh_end = idx_lh_start + N_HAND_LANDMARKS
                left_hand_kps_part = augmented_points[idx_lh_start:idx_lh_end]
                left_hand_valid_mask = np.any(left_hand_kps_part != 0, axis=1)
                if np.any(left_hand_valid_mask):
                    left_hand_kps_part[left_hand_valid_mask, 0] += dx_wrist_l
                    left_hand_kps_part[left_hand_valid_mask, 1] += dy_wrist_l
                augmented_points[idx_lh_start:idx_lh_end] = left_hand_kps_part


        if right_arm_key_points_valid:
            len_r_upper = np.linalg.norm(e_r_orig_xy - s_r_orig_xy)
            len_r_forearm = np.linalg.norm(w_r_orig_xy - e_r_orig_xy)
            w_r_target_xy_for_ik = np.array([w_r_target_x, w_r_orig_xy[1]])

            e_r_new_xy = solve_2d_arm_ik(s_r_orig_xy, w_r_target_xy_for_ik, len_r_upper, len_r_forearm, e_r_orig_xy, w_r_orig_xy)

            if e_r_new_xy is not None:
                dx_wrist_r = w_r_target_xy_for_ik[0] - w_r_orig_xy[0]
                dy_wrist_r = w_r_target_xy_for_ik[1] - w_r_orig_xy[1]

                augmented_points[POSE_LM_RIGHT_ELBOW, 0:2] = e_r_new_xy
                augmented_points[POSE_LM_RIGHT_WRIST, 0:2] = w_r_target_xy_for_ik

                idx_rh_start = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS
                idx_rh_end = idx_rh_start + N_HAND_LANDMARKS
                right_hand_kps_part = augmented_points[idx_rh_start:idx_rh_end]
                right_hand_valid_mask = np.any(right_hand_kps_part != 0, axis=1)
                if np.any(right_hand_valid_mask):
                    right_hand_kps_part[right_hand_valid_mask, 0] += dx_wrist_r
                    right_hand_kps_part[right_hand_valid_mask, 1] += dy_wrist_r
                augmented_points[idx_rh_start:idx_rh_end] = right_hand_kps_part


        if abs(current_overall_dy_shift) > 1e-5:
            arm_and_hand_indices = [
                POSE_LM_LEFT_WRIST, POSE_LM_LEFT_ELBOW,
                POSE_LM_RIGHT_WRIST, POSE_LM_RIGHT_ELBOW
            ]
            arm_and_hand_indices.extend(list(range(N_UPPER_BODY_POSE_LANDMARKS, num_total_landmarks)))
            unique_arm_hand_indices = sorted(list(set(arm_and_hand_indices)))

            for idx in unique_arm_hand_indices:
                is_left_arm_part = (idx == POSE_LM_LEFT_WRIST or idx == POSE_LM_LEFT_ELBOW)
                is_right_arm_part = (idx == POSE_LM_RIGHT_WRIST or idx == POSE_LM_RIGHT_ELBOW)
                is_left_hand_part = (N_UPPER_BODY_POSE_LANDMARKS <= idx < N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS)
                is_right_hand_part = (N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS <= idx < num_total_landmarks)

                should_shift_y = ((is_left_arm_part and left_arm_key_points_valid) or
                                  (is_right_arm_part and right_arm_key_points_valid) or
                                  (is_left_hand_part and left_arm_key_points_valid) or
                                  (is_right_hand_part and right_arm_key_points_valid))

                if should_shift_y and idx < len(augmented_points) and np.any(augmented_points[idx, 0:2] != 0):
                    augmented_points[idx, 1] += current_overall_dy_shift


        if clip_to_01:
            indices_to_clip = list(range(POSE_LM_LEFT_SHOULDER, num_total_landmarks))
            for idx in indices_to_clip:
                 if idx < len(augmented_points) and np.any(augmented_points[idx, 0:2] != 0):
                    augmented_points[idx, 0] = np.clip(augmented_points[idx, 0], 0.0, 1.0)
                    augmented_points[idx, 1] = np.clip(augmented_points[idx, 1], 0.0, 1.0)

        augmented_frame_flat_output = augmented_points.flatten()
        if np.isnan(augmented_frame_flat_output).any() or np.isinf(augmented_frame_flat_output).any():
            augmented_sequence.append(frame_keypoints_flat.copy())
        else:
            augmented_sequence.append(augmented_frame_flat_output)
    return augmented_sequence