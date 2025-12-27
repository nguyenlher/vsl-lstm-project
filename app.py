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
from PIL import Image, ImageDraw, ImageFont

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

def put_vietnamese_text(image, text, position, font_size=30, color=(0, 255, 0)):
    """
    Display Vietnamese text on image using PIL
    
    Args:
        image: OpenCV image (BGR)
        text: Text to display (supports Vietnamese)
        position: (x, y) tuple for text position
        font_size: Font size
        color: RGB color tuple
    
    Returns:
        Image with text overlay
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to use a font that supports Vietnamese, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Draw text (PIL uses RGB, so convert color)
    draw.text(position, text, font=font, fill=color)
    
    # Convert back to BGR for OpenCV
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image_bgr

def normalize_sequence(sequence):
    """
    Normalize keypoint sequence using MIN-MAX normalization.
    This MUST match the normalization used during training (augmentations.py).
    
    Training pipeline uses normalize_to_01=True which normalizes x,y to [0,1].
    
    Args:
        sequence: List of flattened keypoint arrays
        
    Returns:
        Normalized sequence with x,y in range [0,1]
    """
    if not sequence:
        return sequence
    
    normalized_sequence = []
    
    for frame_flat in sequence:
        if frame_flat is None:
            normalized_sequence.append(None)
            continue
        
        # Reshape to (67, 3): [25 pose + 21 left_hand + 21 right_hand]
        kp_3d = frame_flat.copy().reshape(-1, 3)
        
        # Find valid keypoints (non-zero x or y)
        valid_mask = np.any(kp_3d[:, :2] != 0, axis=1)
        
        if np.any(valid_mask):
            x_coords = kp_3d[valid_mask, 0]
            y_coords = kp_3d[valid_mask, 1]
            
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            
            # Normalize X to [0, 1]
            if (max_x - min_x) > 1e-7:
                kp_3d[valid_mask, 0] = (x_coords - min_x) / (max_x - min_x)
            elif x_coords.size > 0:
                kp_3d[valid_mask, 0] = 0.5
            
            # Normalize Y to [0, 1]
            if (max_y - min_y) > 1e-7:
                kp_3d[valid_mask, 1] = (y_coords - min_y) / (max_y - min_y)
            elif y_coords.size > 0:
                kp_3d[valid_mask, 1] = 0.5
        
        normalized_sequence.append(kp_3d.flatten())
    
    return normalized_sequence


def smooth_keypoints(sequence, window_size=3):
    """
    Apply simple moving average to smooth noisy keypoints from webcam.
    This helps reduce jitter and improves prediction stability for real-time input.
    
    Args:
        sequence: List of flattened keypoint arrays
        window_size: Number of frames to average (default: 3)
        
    Returns:
        Smoothed keypoint sequence
    """
    if len(sequence) < window_size:
        return sequence
    
    smoothed = []
    for i in range(len(sequence)):
        start = max(0, i - window_size // 2)
        end = min(len(sequence), i + window_size // 2 + 1)
        window = [seq for seq in sequence[start:end] if seq is not None]
        if window:
            smoothed.append(np.mean(window, axis=0))
        else:
            smoothed.append(sequence[i])
    return smoothed


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


def extract_sequence_from_video(video_path, holistic, max_frames=100):
    """Extract keypoint sequence from video file.
    
    Args:
        video_path: Path to video file
        holistic: MediaPipe holistic model
        max_frames: Maximum number of frames to sample (default: 100, matching dataset preprocessing)
    
    Returns:
        List of keypoint arrays
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total == 0:
        cap.release()
        return []
    
    # Use step-based sampling to match training preprocessing
    step = max(1, total // max_frames)

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
# PROCESS VIDEO AND EXTRACT FRAMES WITH LANDMARKS
# =========================
def extract_landmark_frames(input_path, holistic, max_display_frames=30):
    """
    Process video và extract frames với landmarks để hiển thị
    Returns: List of processed frames (RGB format for Streamlit)
    """
    cap = cv2.VideoCapture(input_path)
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total == 0:
        cap.release()
        return []
    
    # Sample frames for display
    frame_step = max(1, total // max_display_frames)
    
    processed_frames = []
    progress = st.progress(0)
    
    frame_idx = 0
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process sampled frames
        if count % frame_step == 0:
            # Process with MediaPipe
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
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
            
            # Convert BGR to RGB for Streamlit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_frames.append(image_rgb)
            frame_idx += 1
        
        count += 1
        if total > 0:
            progress.progress(min(count / total, 1.0))
    
    cap.release()
    progress.empty()
    
    return processed_frames

# =========================
# UI
# =========================
st.title("Vietnamese Sign Language Recognition")
st.markdown("---")

with st.sidebar:
    st.header("Settings")
    input_mode = st.radio("Input source", ["Webcam", "Video file"], index=0)
    min_det_conf = st.slider("Detection confidence", 0.1, 1.0, 0.5)
    min_track_conf = st.slider("Tracking confidence", 0.1, 1.0, 0.5)

holistic = mp_holistic.Holistic(
    min_detection_confidence=min_det_conf,
    min_tracking_confidence=min_track_conf
)

sequence = None

# =========================
# VIDEO FILE MODE - IMPROVED UI WITH SIDE-BY-SIDE DISPLAY
# =========================
if input_mode == "Video file":
    uploaded = st.file_uploader("Upload video (.mp4)", type=["mp4", "avi"])

    if uploaded:
        # Get unique file ID to detect when a new file is uploaded
        file_id = f"{uploaded.name}_{uploaded.size}"
        
        # Initialize session state for caching
        if 'cached_file_id' not in st.session_state:
            st.session_state.cached_file_id = None
        if 'cached_landmark_frames' not in st.session_state:
            st.session_state.cached_landmark_frames = None
        if 'cached_video_path' not in st.session_state:
            st.session_state.cached_video_path = None
        if 'cached_sequence' not in st.session_state:
            st.session_state.cached_sequence = None
        
        # Check if this is a new file upload
        is_new_file = (st.session_state.cached_file_id != file_id)
        
        if is_new_file:
            # Save video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                tmp.flush()
                os.fsync(tmp.fileno())
                video_path = tmp.name
            
            # Process video and extract landmark frames (only once)
            with st.spinner("Processing video with landmarks..."):
                landmark_frames = extract_landmark_frames(video_path, holistic)
            
            # Automatically extract sequence for prediction
            with st.spinner("Extracting sequence for prediction..."):
                sequence = extract_sequence_from_video(video_path, holistic)
            
            # Cache the results
            st.session_state.cached_file_id = file_id
            st.session_state.cached_landmark_frames = landmark_frames
            st.session_state.cached_video_path = video_path
            st.session_state.cached_sequence = sequence
        else:
            # Use cached results
            landmark_frames = st.session_state.cached_landmark_frames
            video_path = st.session_state.cached_video_path
            sequence = st.session_state.cached_sequence
        
        st.subheader("Video Comparison")
        
        if landmark_frames:
            # Create 2 columns for side-by-side display with equal width
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("##### Original Video")
                st.video(uploaded)
            
            with col2:
                # Frame selector - slider at the top
                selected_frame_idx = st.slider(
                    label="frame_slider",
                    min_value=0,
                    max_value=len(landmark_frames) - 1,
                    value=0,
                    key="frame_selector",
                    label_visibility="collapsed"
                )
                
                # Display selected frame
                st.image(
                    landmark_frames[selected_frame_idx],
                    use_container_width=True
                )
        else:
            st.error("Không thể xử lý video. Vui lòng thử lại.")
            sequence = None



# =========================
# WEBCAM MODE - automatic recognition
# =========================
elif input_mode == "Webcam":
    st.info("Automatic recognition when sign language is detected. Show hand gestures to start recording.")
    
    # Initialize session state for webcam control
    if 'auto_webcam_running' not in st.session_state:
        st.session_state.auto_webcam_running = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Auto Recognition", type="primary", disabled=st.session_state.auto_webcam_running):
            st.session_state.auto_webcam_running = True
            st.rerun()
    
    with col2:
        if st.button("Stop", type="secondary", disabled=not st.session_state.auto_webcam_running):
            st.session_state.auto_webcam_running = False
            st.rerun()
    
    # Display webcam feed and auto predictions if running
    if st.session_state.auto_webcam_running:
        st.markdown("---")
        
        # Create columns to center and reduce webcam size
        _, center_col, _ = st.columns([0.5, 2, 0.5])
        
        with center_col:
            # Create placeholders for video
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        
        # Initialize current prediction in session state
        if 'current_prediction' not in st.session_state:
            st.session_state.current_prediction = None
            st.session_state.current_confidence = 0
        
        try:
            sequence = []
            is_recording = False
            no_hand_frames = 0
            recording_start = None
            
            # Thresholds
            NO_HAND_THRESHOLD = 15  # Number of frames without hands to stop recording
            MIN_SEQUENCE_LENGTH = 10  # Minimum frames to make a prediction
            MAX_RECORDING_TIME = 5.0  # Maximum recording time in seconds
            
            while st.session_state.auto_webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot access webcam. Please check your camera connection.")
                    st.session_state.auto_webcam_running = False
                    break
                
                # Process frame with MediaPipe
                image, results = mediapipe_detection(frame, holistic)
                
                # Draw landmarks
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
                
                # Check if hands are detected
                hands_detected = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
                
                # State machine for recording
                if hands_detected:
                    if not is_recording:
                        # Start recording
                        is_recording = True
                        recording_start = time.time()
                        sequence = []
                        no_hand_frames = 0
                    
                    # Collect keypoints while recording
                    sequence.append(extract_keypoints(results))
                    no_hand_frames = 0
                else:
                    if is_recording:
                        no_hand_frames += 1
                
                # Check if we should stop recording and make prediction
                if is_recording:
                    elapsed = time.time() - recording_start
                    
                    # Stop conditions
                    should_stop = (
                        no_hand_frames >= NO_HAND_THRESHOLD or  # No hands detected for threshold frames
                        elapsed >= MAX_RECORDING_TIME  # Max recording time exceeded
                    )
                    
                    if should_stop and len(sequence) >= MIN_SEQUENCE_LENGTH:
                        # Make prediction - Apply smoothing for webcam to reduce jitter
                        smoothed_sequence = smooth_keypoints(sequence, window_size=3)
                        normalized_sequence = normalize_sequence(smoothed_sequence)
                        kp = interpolate_keypoints(normalized_sequence)
                        
                        if kp is not None:
                            probs = model.predict(kp[None, ...], verbose=0)[0]
                            top_idx = np.argmax(probs)
                            
                            # Update current prediction in session state
                            st.session_state.current_prediction = inv_label_map[top_idx]
                            st.session_state.current_confidence = probs[top_idx] * 100
                        
                        # Reset recording state
                        is_recording = False
                        sequence = []
                        no_hand_frames = 0
                    elif should_stop:
                        # Sequence too short, just reset
                        is_recording = False
                        sequence = []
                        no_hand_frames = 0
                
                # Display current prediction on video
                if st.session_state.current_prediction:
                    pred_text = f"{st.session_state.current_prediction}: {st.session_state.current_confidence:.1f}%"
                    image = put_vietnamese_text(image, pred_text, (10, 30), font_size=30, color=(0, 255, 0))
                
                # Display recording status
                if is_recording:
                    recording_time = time.time() - recording_start
                    status_text = f"Predicting... {recording_time:.1f}s"
                    image = put_vietnamese_text(image, status_text, (10, 70), font_size=25, color=(0, 0, 255))
                
                # Display frame in center column
                with center_col:
                    frame_placeholder.image(image, channels="BGR", use_container_width=True)
                    
                    # Status info
                    if is_recording:
                        status_placeholder.info(f"Recording sign language... {len(sequence)} frames captured")
                    elif hands_detected:
                        status_placeholder.warning("Hands detected - Start performing sign language")
                    else:
                        status_placeholder.success("Waiting for sign language gesture...")
                
                # Small delay to control frame rate
                time.sleep(0.03)  # ~30 FPS
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.auto_webcam_running = False
        finally:
            cap.release()
            if not st.session_state.auto_webcam_running:
                frame_placeholder.empty()
                timer_placeholder.success("Auto recognition session ended")






# =========================
# PREDICTION
# =========================
if sequence:
    st.markdown("---")
    st.header("Prediction Result")
    
    normalized_sequence = normalize_sequence(sequence)
    
    kp = interpolate_keypoints(normalized_sequence)
    if kp is not None:
        probs = model.predict(kp[None, ...])[0]

        # Get the prediction with highest confidence
        top_idx = np.argmax(probs)
        
        # Display result in simple format
        st.write(f"**{inv_label_map[top_idx]}** — {probs[top_idx]*100:.2f}%")
    else:
        st.error("Không thể xử lý sequence. Vui lòng thử lại.")
