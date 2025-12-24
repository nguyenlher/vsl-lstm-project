import cv2
import numpy as np
import os
import re
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from scipy.interpolate import interp1d
import random
import gc
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from augmentation import (
    augment_inter_hand_distance, 
    augment_keypoints_scaling,
    augment_keypoints_rotation,
    augment_keypoints_translation,
    augment_keypoints_time_stretch
)

# ==================== CẤU HÌNH ====================
CONFIG = {
    'NUM_AUGMENTED_SAMPLES': 10,
    'MAX_AUGS_PER_SAMPLE': 3,
    'SEQUENCE_LENGTH': 60,
    'MIN_DETECTION_CONFIDENCE': 0.5,
    'MIN_TRACKING_CONFIDENCE': 0.5,
    'NUM_WORKERS': max(1, min(4, cpu_count() - 1)),
}

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('augmentation.log'),
        logging.StreamHandler()
    ]
)

mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS

# --- GIỮ NGUYÊN CÁC HÀM HELPER CƠ BẢN ---
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose_kps = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    if results and results.pose_landmarks:
        for i in range(N_UPPER_BODY_POSE_LANDMARKS):
            if i < len(results.pose_landmarks.landmark):
                res = results.pose_landmarks.landmark[i]
                pose_kps[i] = [res.x, res.y, res.z]
    if results and results.left_hand_landmarks:
        left_hand_kps = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    if results and results.right_hand_landmarks:
        right_hand_kps = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    keypoints = np.concatenate([pose_kps,left_hand_kps, right_hand_kps])
    return keypoints.flatten()

def interpolate_keypoints(keypoints_sequence, target_len = 60):
    if len(keypoints_sequence) == 0:
        return None
    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_len)
    num_features = keypoints_sequence[0].shape[0]
    interpolated_sequence = np.zeros((target_len, num_features))
    for feature_idx in range(num_features):
        feature_values = [frame[feature_idx] for frame in keypoints_sequence]
        interpolator = interp1d(
            original_times, feature_values,
            kind='cubic', bounds_error=False, fill_value="extrapolate"
        )
        interpolated_sequence[:, feature_idx] = interpolator(target_times)
    return interpolated_sequence

def sequence_frames(video_path, holistic, max_frames=100):
    sequence_frames_list = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning(f"Cannot open video {video_path}")
        return sequence_frames_list
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return sequence_frames_list
    step = max(1, total_frames // max_frames)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % step != 0:
            frame_idx += 1
            continue
        try:
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            if keypoints is not None:
                sequence_frames_list.append(keypoints)
        except Exception as e:
            logging.warning(f"Error processing frame {frame_idx}: {e}")
        frame_idx += 1
    cap.release()
    return sequence_frames_list

class GetTime():
    def __init__(self):
        self.starttime = datetime.now()
    def get_time(self):
        return datetime.now() - self.starttime

augmentations = [
    augment_keypoints_scaling,
    augment_keypoints_rotation,
    augment_keypoints_translation,
    augment_keypoints_time_stretch,
    augment_inter_hand_distance
]

def generate_augmented_samples(original_sequence, augmentation_functions, num_samples_to_generate: int, max_augs_per_sample: int = 3):
    if not original_sequence or not augmentation_functions:
        return
    num_available_augs = len(augmentation_functions)
    for i in range(num_samples_to_generate):
        current_sequence = [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in original_sequence]
        num_augs_to_apply = random.randint(1, min(max_augs_per_sample, num_available_augs))
        selected_aug_funcs = random.sample(augmentation_functions, num_augs_to_apply)
        random.shuffle(selected_aug_funcs)
        is_valid = True
        for aug_func in selected_aug_funcs:
            try:
                current_sequence = aug_func(current_sequence)
                if not current_sequence or all(frame is None for frame in current_sequence):
                    is_valid = False
                    break
            except Exception as e:
                # logging.warning(f"Augmentation {aug_func.__name__} failed: {e}")
                is_valid = False
                break
        if is_valid and current_sequence:
            yield current_sequence

def save_sample(file_path, sequence, label):
    try:
        np.savez(file_path, sequence=sequence, label=label)
        return True
    except Exception as e:
        logging.error(f"Error saving {file_path}: {e}")
        return False

# ==================== PROCESS LOGIC (UPDATED FOR FLAT STRUCTURE) ====================

def process_single_video(args):
    """Xử lý một video duy nhất - Flat Structure Version."""
    row_data, label_map, DATA_PATH, video_folder, sequence_length = args
    
    action = row_data['LABEL']
    video_file = row_data['VIDEO'] # Ví dụ: "D001.mp4"
    video_name_base = os.path.splitext(video_file)[0] # "D001"
    
    label = label_map[action]
    
    # --- LOGIC SKIP MỚI CHO FLAT STRUCTURE ---
    # Kiểm tra xem file gốc đã tồn tại chưa. Nếu có rồi nghĩa là video này đã xử lý.
    # Đường dẫn file gốc: dataset/augmented/D001_orig.npz
    expected_orig_path = os.path.join(DATA_PATH, f"{video_name_base}_orig.npz")
    
    if os.path.exists(expected_orig_path):
        return {'video': video_file, 'status': 'skipped', 'samples': 0}
    
    video_path = os.path.join(video_folder, video_file)
    if not os.path.exists(video_path):
        return {'video': video_file, 'status': 'error', 'message': 'Video not found'}
    
    holistic_config = {
        'min_detection_confidence': CONFIG['MIN_DETECTION_CONFIDENCE'],
        'min_tracking_confidence': CONFIG['MIN_TRACKING_CONFIDENCE']
    }
    
    try:
        with mp_holistic.Holistic(**holistic_config) as holistic:
            frame_lists = sequence_frames(video_path, holistic)
            
            if not frame_lists:
                return {'video': video_file, 'status': 'error', 'message': 'No frames extracted'}
            
            idx = 0
            
            # 1. Lưu sequence gốc
            # Tên file: VideoName_orig.npz
            original_seq = interpolate_keypoints(frame_lists, sequence_length)
            if original_seq is not None:
                file_path = os.path.join(DATA_PATH, f'{video_name_base}_orig.npz')
                save_sample(file_path, original_seq, label)
                idx += 1
            
            # 2. Tạo và lưu các mẫu augmented
            # Tên file: VideoName_aug_0.npz, VideoName_aug_1.npz...
            for aug_seq in generate_augmented_samples(
                frame_lists, 
                augmentations, 
                CONFIG['NUM_AUGMENTED_SAMPLES'],
                CONFIG['MAX_AUGS_PER_SAMPLE']
            ):
                interpolated = interpolate_keypoints(aug_seq, sequence_length)
                if interpolated is not None:
                    file_path = os.path.join(DATA_PATH, f'{video_name_base}_aug_{idx-1}.npz')
                    save_sample(file_path, interpolated, label)
                    idx += 1
            
            del frame_lists
            gc.collect()
            
            return {'video': video_file, 'status': 'success', 'samples': idx}
    
    except Exception as e:
        return {'video': video_file, 'status': 'error', 'message': str(e)}

def main():
    # ==================== ĐƯỜNG DẪN ====================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Flat Structure: Tất cả ném vào augmented/, không chia folder con
    DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'augmented')
    DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
    LOG_PATH = os.path.join(BASE_DIR, DATASET_PATH, 'label')

    sequence_length = CONFIG['SEQUENCE_LENGTH']

    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    label_file = os.path.join(DATASET_PATH, 'label', 'label.csv')
    video_folder = os.path.join(DATASET_PATH, 'videos')

    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
    if not os.path.isdir(video_folder):
        raise FileNotFoundError(f"Video folder not found: {video_folder}")

    df = pd.read_csv(label_file)
    required_columns = ['LABEL', 'VIDEO']
    if any(col not in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns in label.csv")

    selected_actions = sorted(df['LABEL'].unique())
    label_map = {action: idx for idx, action in enumerate(selected_actions)}

    # Vẫn lưu label_map để sau này training biết số 0 là gì, số 1 là gì
    label_map_path = os.path.join(LOG_PATH, 'label_map.json')
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)

    logging.info(f"Selected {len(selected_actions)} actions.")
    logging.info(f"Output Directory (Flat Structure): {DATA_PATH}")
    logging.info(f"Using {CONFIG['NUM_WORKERS']} workers")

    time_tracker = GetTime()
    logging.info("Start processing data...")

    process_args = [
        (row, label_map, DATA_PATH, video_folder, sequence_length)
        for _, row in df.iterrows()
    ]
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    with Pool(processes=CONFIG['NUM_WORKERS']) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_video, process_args),
            total=len(process_args),
            desc='Processing videos'
        ))
    
    for result in results:
        if result['status'] == 'success':
            total_processed += 1
        elif result['status'] == 'skipped':
            total_skipped += 1
        elif result['status'] == 'error':
            total_errors += 1
            logging.warning(f"Error processing {result['video']}: {result.get('message', 'Unknown error')}")
    
    logging.info("="*50)
    logging.info("DATA PROCESSING COMPLETED.")
    logging.info(f"Total videos: {len(df)}")
    logging.info(f"Processed: {total_processed}")
    logging.info(f"Skipped: {total_skipped}")
    logging.info(f"Errors: {total_errors}")
    logging.info(f"Total time: {time_tracker.get_time()}")
    logging.info("="*50)

if __name__ == '__main__':
    main()