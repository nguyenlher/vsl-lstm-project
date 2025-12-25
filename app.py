import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile
import os
import cv2
import mediapipe as mp
import json
import time
from scipy.interpolate import interp1d

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Vietnamese Sign Language Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# MEDIAPIPE SETUP
# =========================
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

N_POSE = 25
N_HAND = 21
N_FEATURES = (N_POSE + 2 * N_HAND) * 3

# =========================
# LOAD MODEL & LABEL MAP
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/final_model.keras")

@st.cache_data
def load_label_map():
    with open("dataset/label/label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map

model = load_model()
label_map, inv_label_map = load_label_map()

# =========================
# MEDIAPIPE FUNCTIONS
# =========================
def mediapipe_detection(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.zeros((N_POSE, 3))
    lh = np.zeros((N_HAND, 3))
    rh = np.zeros((N_HAND, 3))

    if results.pose_landmarks:
        for i in range(min(N_POSE, len(results.pose_landmarks.landmark))):
            lm = results.pose_landmarks.landmark[i]
            pose[i] = [lm.x, lm.y, lm.z]

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

    return np.concatenate([pose, lh, rh]).flatten()

# =========================
# SEQUENCE PROCESSING
# =========================
def interpolate_keypoints(sequence, target_len=60):
    if len(sequence) == 0:
        return None

    seq = np.array(sequence)
    t_original = np.linspace(0, 1, len(seq))
    t_target = np.linspace(0, 1, target_len)

    interpolated = np.zeros((target_len, seq.shape[1]))
    for i in range(seq.shape[1]):
        f = interp1d(t_original, seq[:, i], kind="cubic", fill_value="extrapolate")
        interpolated[:, i] = f(t_target)

    return interpolated

def extract_sequence_from_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // 80)

    sequence = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            _, results = mediapipe_detection(frame, holistic)
            sequence.append(extract_keypoints(results))

        idx += 1

    cap.release()
    return sequence

# =========================
# PROCESS VIDEO AND SHOW LANDMARKS
# =========================
def process_and_display_video(input_path, holistic):
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0 or fps is None:
        fps = 25

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Lấy mỗi N frame để hiển thị (không cần xử lý hết tất cả frame)
    frame_step = max(1, total // 30)  # Lấy khoảng 30 frame
    
    progress = st.progress(0)
    processed_frames = []
    count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chỉ xử lý một số frame nhất định
        if count % frame_step == 0:
            image, results = mediapipe_detection(frame, holistic)
            
            # Vẽ landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Chuyển BGR sang RGB để hiển thị đúng màu
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_frames.append(image_rgb)
            frame_idx += 1
        
        count += 1
        if total > 0:
            progress.progress(min(count / total, 1.0))

    cap.release()
    progress.empty()

    return processed_frames, fps

# =========================
# UI
# =========================
st.title("🇻🇳 Vietnamese Sign Language Recognition")
st.markdown("---")

with st.sidebar:
    st.header("Settings")
    input_mode = st.radio("Input source", ["Video file", "Webcam"])
    min_det_conf = st.slider("Detection confidence", 0.1, 1.0, 0.5)
    min_track_conf = st.slider("Tracking confidence", 0.1, 1.0, 0.5)

holistic = mp_holistic.Holistic(
    min_detection_confidence=min_det_conf,
    min_tracking_confidence=min_track_conf
)

sequence = None

# =========================
# VIDEO FILE MODE
# =========================
if input_mode == "Video file":
    uploaded = st.file_uploader("Upload video (.mp4)", type=["mp4", "avi"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp.flush()
            os.fsync(tmp.fileno())
            video_path = tmp.name

        # Hiển thị video gốc
        st.subheader("Original Video")
        st.video(uploaded)
        
        st.subheader("🎯 Video with landmarks")
        
        # Xử lý và lấy các frame
        processed_frames, fps = process_and_display_video(video_path, holistic)
        
        if len(processed_frames) > 0:
            # Tạo slideshow với st.image và animation
            st.info(f"Hiển thị {len(processed_frames)} frames với landmarks")
            
            # Thêm slider để người dùng có thể xem từng frame
            frame_index = st.slider(
                "Chọn frame để xem", 
                0, 
                len(processed_frames) - 1, 
                0
            )
            
            # Hiển thị frame được chọn
            st.image(processed_frames[frame_index], use_container_width=True)
            
            # Tùy chọn: Auto-play slideshow
            if st.checkbox("Auto-play slideshow"):
                placeholder = st.empty()
                delay = 1.0 / fps if fps > 0 else 0.04  # Tính delay dựa trên FPS
                
                for idx, frame in enumerate(processed_frames):
                    placeholder.image(frame, use_container_width=True, caption=f"Frame {idx+1}/{len(processed_frames)}")
                    time.sleep(delay)
                
                placeholder.empty()
                st.success("Slideshow completed!")
        else:
            st.error("Không thể xử lý video. Vui lòng thử lại.")

        if st.button("Predict", type="primary"):
            with st.spinner("Processing video..."):
                sequence = extract_sequence_from_video(video_path, holistic)

# =========================
# WEBCAM MODE
# =========================
else:
    st.info("Record webcam for 4 seconds")

    if st.button("Record & Predict", type="primary"):
        cap = cv2.VideoCapture(0)
        start = time.time()
        sequence = []
        frame_box = st.empty()

        while time.time() - start < 4:
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            sequence.append(extract_keypoints(results))
            frame_box.image(image, channels="BGR")

        cap.release()

# =========================
# PREDICTION
# =========================
if sequence:
    st.markdown("---")
    st.header("📊 Prediction Result")

    kp = interpolate_keypoints(sequence)
    if kp is not None:
        probs = model.predict(kp[None, ...])[0]

        top3 = np.argsort(probs)[-3:][::-1]

        for i, idx in enumerate(top3, 1):
            st.write(f"**{i}. {inv_label_map[idx]}** — {probs[idx]*100:.2f}%")

        st.success(f"Best prediction: **{inv_label_map[top3[0]]}**")
    else:
        st.error("Không thể xử lý sequence. Vui lòng thử lại.")