import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from scipy.interpolate import interp1d
import json

# Load model và label_map
model = tf.keras.models.load_model('models/best_model.keras')
with open('dataset/label/label_map.json', 'r', encoding='utf-8') as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 33
N_FACE_LANDMARKS = 11
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_FACE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose_kps = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    face_kps = np.zeros((N_FACE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    if results and results.pose_landmarks:
        for i in range(N_UPPER_BODY_POSE_LANDMARKS):
            if i < len(results.pose_landmarks.landmark):
                res = results.pose_landmarks.landmark[i]
                pose_kps[i] = [res.x, res.y, res.z]
    if results and results.face_landmarks:
        for i in range(N_FACE_LANDMARKS):
            if i < len(results.face_landmarks.landmark):
                res = results.face_landmarks.landmark[i]
                face_kps[i] = [res.x, res.y, res.z]
    if results and results.left_hand_landmarks:
        left_hand_kps = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    if results and results.right_hand_landmarks:
        right_hand_kps = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    keypoints = np.concatenate([pose_kps, face_kps, left_hand_kps, right_hand_kps])
    return keypoints.flatten()

def interpolate_keypoints(keypoints_sequence, target_len=90):
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
        print(f"Cannot open video {video_path}")
        return sequence_frames_list
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return sequence_frames_list
    step = max(1, total_frames // 120)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
            continue

        try:
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            if keypoints is not None:
                sequence_frames_list.append(keypoints)
        except Exception as e:
            continue

    cap.release()
    return sequence_frames_list

# Test trên video
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
video_path = 'dataset/videos/D0001B.mp4'
sequence = sequence_frames(video_path, holistic)
if sequence:
    kp = interpolate_keypoints(sequence)
    result = model.predict(np.expand_dims(kp, axis=0))
    pred_idx = np.argmax(result, axis=1)
    pred_label = [inv_label_map[idx] for idx in pred_idx]
    print(f"Predicted label: {pred_label}")
else:
    print("No sequence extracted")