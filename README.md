# Vietnamese Sign Language Recognition

Real-time Vietnamese sign language recognition using MediaPipe Holistic and BiLSTM.

## Features

- Dataset 3315+ Vietnamese sign language from QIPEDC
- Real-time webcam inference with PyQt5 GUI

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Download dataset
python download_dataset.py

# Generate augmented data
python run_augmentation.py

# Train model
jupyter notebook training.ipynb

# Run desktop app
python app.py
```

## Project Structure

```
vsl-lstm-project/
├── app.py                 # PyQt5 desktop app
├── training.ipynb         # Model training
├── run_augmentation.py    # Data augmentation
├── download_dataset.py    # Dataset scraper
├── augmentation/          # Augmentation modules
├── dataset/
│   ├── videos/            # Raw videos
│   ├── augmented/         # NPZ keypoints
│   └── label/             # Labels
└── models/
    └── final_model.keras  # Trained model
```

## Requirements

Python 3.8+, TensorFlow 2.10+, MediaPipe, OpenCV, PyQt5

## License

Educational use only. Dataset: [QIPEDC](https://qipedc.moet.gov.vn)
