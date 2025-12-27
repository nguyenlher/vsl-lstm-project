import numpy as np
import logging
from typing import Optional, Tuple

def compute_ik_elbow_solutions(
    p_shoulder_xy: np.ndarray,
    p_wrist_target_xy: np.ndarray,
    len_upper_arm: float,
    len_forearm: float
) -> Tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(p_wrist_target_xy - p_shoulder_xy)
    if d < 1e-9:
        d = 1e-9

    a = (len_upper_arm**2 - len_forearm**2 + d**2) / (2 * d)
    h_squared = len_upper_arm**2 - a**2

    if h_squared < -1e-9:
        logging.error(f"IK Critical Error: h_squared is significantly negative ({h_squared:.4g}). d={d:.3f}, l1={len_upper_arm:.3f}, l2={len_forearm:.3f}, a={a:.3f}")
        vec_sw = (p_wrist_target_xy - p_shoulder_xy) / d
        return p_shoulder_xy + vec_sw * len_upper_arm, p_shoulder_xy + vec_sw * len_upper_arm
    h = np.sqrt(max(0, h_squared))

    p2_x = p_shoulder_xy[0] + a * (p_wrist_target_xy[0] - p_shoulder_xy[0]) / d
    p2_y = p_shoulder_xy[1] + a * (p_wrist_target_xy[1] - p_shoulder_xy[1]) / d

    perp_vec_x = -(p_wrist_target_xy[1] - p_shoulder_xy[1]) / d
    perp_vec_y = (p_wrist_target_xy[0] - p_shoulder_xy[0]) / d

    elbow_sol1_xy = np.array([p2_x + h * perp_vec_x, p2_y + h * perp_vec_y])
    elbow_sol2_xy = np.array([p2_x - h * perp_vec_x, p2_y - h * perp_vec_y])

    return elbow_sol1_xy, elbow_sol2_xy


def select_ik_elbow_solution(
    elbow_sol1_xy: np.ndarray,
    elbow_sol2_xy: np.ndarray,
    original_elbow_xy: Optional[np.ndarray],
    original_wrist_xy: Optional[np.ndarray],
    p_shoulder_xy: np.ndarray,
    p_wrist_target_xy: np.ndarray,
    prefer_original_bend: bool
) -> np.ndarray:
    if not prefer_original_bend or original_elbow_xy is None or original_wrist_xy is None:
        if original_elbow_xy is not None:
            dist1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
            dist2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
            return elbow_sol1_xy if dist1 <= dist2 else elbow_sol2_xy
        return elbow_sol1_xy

    vec_sw_orig = original_wrist_xy - p_shoulder_xy
    if np.linalg.norm(vec_sw_orig) < 1e-5:
        dist1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
        dist2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
        return elbow_sol1_xy if dist1 <= dist2 else elbow_sol2_xy

    original_side = ((original_wrist_xy[0] - p_shoulder_xy[0]) * (original_elbow_xy[1] - p_shoulder_xy[1]) -
                     (original_wrist_xy[1] - p_shoulder_xy[1]) * (original_elbow_xy[0] - p_shoulder_xy[0]))

    side1 = ((p_wrist_target_xy[0] - p_shoulder_xy[0]) * (elbow_sol1_xy[1] - p_shoulder_xy[1]) -
             (p_wrist_target_xy[1] - p_shoulder_xy[1]) * (elbow_sol1_xy[0] - p_shoulder_xy[0]))
    side2 = ((p_wrist_target_xy[0] - p_shoulder_xy[0]) * (elbow_sol2_xy[1] - p_shoulder_xy[1]) -
             (p_wrist_target_xy[1] - p_shoulder_xy[1]) * (elbow_sol2_xy[0] - p_shoulder_xy[0]))

    if abs(original_side) < 1e-3:
        dist1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
        dist2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
        return elbow_sol1_xy if dist1 <= dist2 else elbow_sol2_xy

    if np.sign(side1) == np.sign(original_side):
        return elbow_sol1_xy
    elif np.sign(side2) == np.sign(original_side):
        return elbow_sol2_xy
    else:
        dist1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
        dist2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
        logging.warning("IK Warning: No solution matched original bend side. Choosing closest.")
        return elbow_sol1_xy if dist1 <= dist2 else elbow_sol2_xy


def solve_2d_arm_ik(
    p_shoulder_xy: np.ndarray,
    p_wrist_target_xy: np.ndarray,
    len_upper_arm: float,
    len_forearm: float,
    original_elbow_xy: Optional[np.ndarray] = None,
    original_wrist_xy: Optional[np.ndarray] = None,
    prefer_original_bend: bool = True
) -> Optional[np.ndarray]:
    d = np.linalg.norm(p_wrist_target_xy - p_shoulder_xy)
    l1 = max(1e-5, len_upper_arm)
    l2 = max(1e-5, len_forearm)


    if d > l1 + l2 - 1e-5:
        if d < 1e-9:
            return p_shoulder_xy + np.array([l1, 0]) if original_elbow_xy is None else original_elbow_xy.copy()
        vec_sw = (p_wrist_target_xy - p_shoulder_xy) / d
        return p_shoulder_xy + vec_sw * l1


    if d < abs(l1 - l2) + 1e-5:
        logging.warning(f"IK Warning: Target too close. d={d:.3f}, l1={l1:.3f}, l2={l2:.3f}")
        if original_elbow_xy is not None:
            return original_elbow_xy.copy()
        if d < 1e-9:
            return p_shoulder_xy + np.array([l1, 0])
        vec_sw = (p_wrist_target_xy - p_shoulder_xy) / d
        return p_shoulder_xy + vec_sw * l1


    elbow_sol1_xy, elbow_sol2_xy = compute_ik_elbow_solutions(p_shoulder_xy, p_wrist_target_xy, l1, l2)
    return select_ik_elbow_solution(elbow_sol1_xy, elbow_sol2_xy, original_elbow_xy, original_wrist_xy,
                                 p_shoulder_xy, p_wrist_target_xy, prefer_original_bend)