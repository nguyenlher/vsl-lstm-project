import sys
import cv2
import numpy as np
import tensorflow as tf
import json
import time
import os
from scipy.interpolate import interp1d
import mediapipe as mp

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QSlider,
    QGroupBox, QStatusBar, QFrame, QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

N_POSE = 25
N_HAND = 21
N_FEATURES = (N_POSE + 2 * N_HAND) * 3
SEQUENCE_LENGTH = 60
MAX_FRAMES = 100


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

def normalize_sequence(sequence):

    if not sequence:
        return sequence
    
    normalized_sequence = []
    
    for frame_flat in sequence:
        if frame_flat is None:
            normalized_sequence.append(None)
            continue
        
        kp_3d = frame_flat.copy().reshape(-1, 3)
        valid_mask = np.any(kp_3d[:, :2] != 0, axis=1)
        
        if np.any(valid_mask):
            x_coords = kp_3d[valid_mask, 0]
            y_coords = kp_3d[valid_mask, 1]
            
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            
            if (max_x - min_x) > 1e-7:
                kp_3d[valid_mask, 0] = (x_coords - min_x) / (max_x - min_x)
            elif x_coords.size > 0:
                kp_3d[valid_mask, 0] = 0.5
            
            if (max_y - min_y) > 1e-7:
                kp_3d[valid_mask, 1] = (y_coords - min_y) / (max_y - min_y)
            elif y_coords.size > 0:
                kp_3d[valid_mask, 1] = 0.5
        
        normalized_sequence.append(kp_3d.flatten())
    
    return normalized_sequence

def smooth_keypoints(sequence, window_size=3):

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

def interpolate_keypoints(sequence, target_len=SEQUENCE_LENGTH):
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

def draw_landmarks(image, results):

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    
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
    return image


class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VSL RECOGNITION SYSTEM")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QTabWidget::pane {
                border: 1px solid #16213e;
                background-color: #1a1a2e;
            }
            QTabBar::tab {
                background-color: #16213e;
                color: #e94560;
                padding: 12px 40px;
                font-size: 14px;
                font-weight: bold;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background-color: #e94560;
                color: white;
            }
            QPushButton {
                background-color: #e94560;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
            QLabel {
                color: #eee;
            }
            QGroupBox {
                color: #e94560;
                font-weight: bold;
                border: 2px solid #16213e;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #16213e;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #e94560;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        

        self.load_resources()
        

        self.mp_holistic = mp.solutions.holistic
        self.holistic = None
        self.detection_confidence = 0.5
        self.tracking_confidence = 0.5
        

        self.webcam_running = False
        self.webcam_recording = False
        self.sequence = []
        self.no_hand_frames = 0
        self.recording_start = None
        self.current_prediction = ""
        self.current_confidence = 0
        

        self.video_path = None
        self.video_cap = None
        

        self.setup_ui()
        

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_webcam_frame)
        
    def load_resources(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "final_model.keras")
            label_path = os.path.join(base_dir, "dataset", "label", "label_map.json")
            
            self.model = tf.keras.models.load_model(model_path)
            
            with open(label_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)
            self.inv_label_map = {v: k for k, v in self.label_map.items()}
            
        except Exception as e:
            print(f"Error loading resources: {e}")
            self.model = None
            self.label_map = {}
            self.inv_label_map = {}
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        

        title_label = QLabel("VLS RECOGNITION SYSTEM")
        title_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title_label.setStyleSheet("color: #e94560; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        

        webcam_tab = QWidget()
        self.setup_webcam_tab(webcam_tab)
        self.tabs.addTab(webcam_tab, "Webcam")
        

        video_tab = QWidget()
        self.setup_video_tab(video_tab)
        self.tabs.addTab(video_tab, "Video File")
        

        settings_group = QGroupBox("Settings")
        settings_layout = QHBoxLayout(settings_group)
        

        det_layout = QVBoxLayout()
        det_label = QLabel(f"Detection Confidence: {self.detection_confidence:.1f}")
        self.det_slider = QSlider(Qt.Horizontal)
        self.det_slider.setRange(10, 100)
        self.det_slider.setValue(50)
        self.det_slider.valueChanged.connect(
            lambda v: self.update_confidence('detection', v, det_label)
        )
        det_layout.addWidget(det_label)
        det_layout.addWidget(self.det_slider)
        settings_layout.addLayout(det_layout)
        

        track_layout = QVBoxLayout()
        track_label = QLabel(f"Tracking Confidence: {self.tracking_confidence:.1f}")
        self.track_slider = QSlider(Qt.Horizontal)
        self.track_slider.setRange(10, 100)
        self.track_slider.setValue(50)
        self.track_slider.valueChanged.connect(
            lambda v: self.update_confidence('tracking', v, track_label)
        )
        track_layout.addWidget(track_label)
        track_layout.addWidget(self.track_slider)
        settings_layout.addLayout(track_layout)
        
        main_layout.addWidget(settings_group)
        

        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("color: #888;")
        
    def setup_webcam_tab(self, tab):
        layout = QVBoxLayout(tab)
        

        self.webcam_label = QLabel()
        self.webcam_label.setMinimumSize(800, 600)
        self.webcam_label.setStyleSheet("background-color: #0f0f23; border-radius: 10px;")
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setText("Click 'Start Webcam' to begin")
        self.webcam_label.setFont(QFont("Segoe UI", 16))
        layout.addWidget(self.webcam_label)
        

        self.prediction_label = QLabel("Prediction: --")
        self.prediction_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.prediction_label.setStyleSheet("color: #4ecca3; padding: 10px;")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prediction_label)
        

        btn_layout = QHBoxLayout()
        
        self.start_webcam_btn = QPushButton("Start Webcam")
        self.start_webcam_btn.clicked.connect(self.toggle_webcam)
        btn_layout.addWidget(self.start_webcam_btn)
        
        layout.addLayout(btn_layout)
        

        self.webcam_status_label = QLabel("Status: Idle")
        self.webcam_status_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self.webcam_status_label)
        
    def setup_video_tab(self, tab):
        layout = QVBoxLayout(tab)
        

        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: #0f0f23; border-radius: 10px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Upload a video file to analyze")
        self.video_label.setFont(QFont("Segoe UI", 16))
        layout.addWidget(self.video_label)
        

        self.video_prediction_label = QLabel("Prediction: --")
        self.video_prediction_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.video_prediction_label.setStyleSheet("color: #4ecca3; padding: 10px;")
        self.video_prediction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_prediction_label)
        

        btn_layout = QHBoxLayout()
        
        self.upload_btn = QPushButton("Upload Video")
        self.upload_btn.clicked.connect(self.upload_video)
        btn_layout.addWidget(self.upload_btn)
        
        self.analyze_btn = QPushButton("Analyze Video")
        self.analyze_btn.clicked.connect(self.analyze_video)
        self.analyze_btn.setEnabled(False)
        btn_layout.addWidget(self.analyze_btn)
        
        layout.addLayout(btn_layout)
        

        self.video_status_label = QLabel("")
        self.video_status_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self.video_status_label)
    
    def update_confidence(self, conf_type, value, label):
        conf = value / 100.0
        if conf_type == 'detection':
            self.detection_confidence = conf
            label.setText(f"Detection Confidence: {conf:.1f}")
        else:
            self.tracking_confidence = conf
            label.setText(f"Tracking Confidence: {conf:.1f}")
        

        if self.webcam_running:
            self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=self.detection_confidence,
                min_tracking_confidence=self.tracking_confidence
            )
    
    def toggle_webcam(self):
        if not self.webcam_running:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.statusBar().showMessage("Error: Cannot access webcam")
            return
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        
        self.webcam_running = True
        self.sequence = []
        self.no_hand_frames = 0
        self.webcam_recording = False
        
        self.start_webcam_btn.setText("Stop Webcam")
        self.timer.start(33)
        self.statusBar().showMessage("Webcam running...")
    
    def stop_webcam(self):
        self.timer.stop()
        self.webcam_running = False
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        if self.holistic:
            self.holistic.close()
            self.holistic = None
        
        self.start_webcam_btn.setText("Start Webcam")
        self.webcam_label.setText("Click 'Start Webcam' to begin")
        self.statusBar().showMessage("Webcam stopped")
    
    def update_webcam_frame(self):
        if not self.webcam_running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        

        frame = cv2.flip(frame, 1)
        

        image, results = mediapipe_detection(frame, self.holistic)
        image = draw_landmarks(image, results)
        

        hands_detected = (results.left_hand_landmarks is not None or 
                         results.right_hand_landmarks is not None)
        

        NO_HAND_THRESHOLD = 15
        MIN_SEQUENCE_LENGTH = 10
        MAX_RECORDING_TIME = 5.0
        
        if hands_detected:
            if not self.webcam_recording:
                self.webcam_recording = True
                self.recording_start = time.time()
                self.sequence = []
                self.no_hand_frames = 0
            
            self.sequence.append(extract_keypoints(results))
            self.no_hand_frames = 0
        else:
            if self.webcam_recording:
                self.no_hand_frames += 1
        

        if self.webcam_recording:
            elapsed = time.time() - self.recording_start
            
            should_stop = (
                self.no_hand_frames >= NO_HAND_THRESHOLD or
                elapsed >= MAX_RECORDING_TIME
            )
            
            if should_stop and len(self.sequence) >= MIN_SEQUENCE_LENGTH:

                smoothed = smooth_keypoints(self.sequence, window_size=3)
                normalized = normalize_sequence(smoothed)
                kp = interpolate_keypoints(normalized)
                
                if kp is not None and self.model is not None:
                    probs = self.model.predict(kp[None, ...], verbose=0)[0]
                    top_idx = np.argmax(probs)
                    
                    self.current_prediction = self.inv_label_map.get(top_idx, "Unknown")
                    self.current_confidence = probs[top_idx] * 100
                    
                    self.prediction_label.setText(
                        f"Prediction: {self.current_prediction} — {self.current_confidence:.1f}%"
                    )
                
                self.webcam_recording = False
                self.sequence = []
                self.no_hand_frames = 0
            elif should_stop:
                self.webcam_recording = False
                self.sequence = []
                self.no_hand_frames = 0
        

        if self.webcam_recording:
            self.webcam_status_label.setText(
                f"Status: Recording... {len(self.sequence)} frames"
            )
            self.webcam_status_label.setStyleSheet("color: #e94560; font-size: 12px;")
        elif hands_detected:
            self.webcam_status_label.setText("Status: Hands detected - Start signing")
            self.webcam_status_label.setStyleSheet("color: #f9a826; font-size: 12px;")
        else:
            self.webcam_status_label.setText("Status: Waiting for gesture...")
            self.webcam_status_label.setStyleSheet("color: #4ecca3; font-size: 12px;")
        

        self.display_frame(image, self.webcam_label)
    
    def display_frame(self, frame, label):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        

        scaled = qt_image.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(QPixmap.fromImage(scaled))
    
    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        
        if file_path:
            self.video_path = file_path
            self.analyze_btn.setEnabled(True)
            self.video_status_label.setText(f"Loaded: {os.path.basename(file_path)}")
            

            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                self.display_frame(frame, self.video_label)
            cap.release()
    
    def analyze_video(self):
        if not self.video_path:
            return
        
        self.analyze_btn.setEnabled(False)
        self.video_status_label.setText("Analyzing video...")
        QApplication.processEvents()
        
        holistic = self.mp_holistic.Holistic(
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        

        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // MAX_FRAMES)
        
        sequence = []
        idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if idx % step == 0:
                image, results = mediapipe_detection(frame, holistic)
                sequence.append(extract_keypoints(results))
                

                image = draw_landmarks(image, results)
                self.display_frame(image, self.video_label)
                QApplication.processEvents()
            
            idx += 1
            
            if idx % 10 == 0:
                progress = min(100, int((idx / total) * 100))
                self.video_status_label.setText(f"Processing: {progress}%")
                QApplication.processEvents()
        
        cap.release()
        holistic.close()
        

        if len(sequence) >= 10:
            normalized = normalize_sequence(sequence)
            kp = interpolate_keypoints(normalized)
            
            if kp is not None and self.model is not None:
                probs = self.model.predict(kp[None, ...], verbose=0)[0]
                top_idx = np.argmax(probs)
                
                prediction = self.inv_label_map.get(top_idx, "Unknown")
                confidence = probs[top_idx] * 100
                
                self.video_prediction_label.setText(
                    f"Prediction: {prediction} — {confidence:.1f}%"
                )
                self.video_status_label.setText("Analysis complete!")
        else:
            self.video_prediction_label.setText("Error: Not enough frames")
        
        self.analyze_btn.setEnabled(True)
    
    def closeEvent(self, event):
        self.stop_webcam()
        event.accept()



def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(26, 26, 46))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(15, 15, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(26, 26, 46))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(26, 26, 46))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Highlight, QColor(233, 69, 96))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    
    window = SignLanguageApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
