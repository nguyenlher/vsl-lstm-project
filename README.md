# Vietnamese Sign Language Recognition

A deep learning project for recognizing Vietnamese Sign Language (VSL) gestures using LSTM neural networks and Mediapipe for keypoint extraction.

## Features

- **Real-time Recognition**: Process webcam input or uploaded videos
- **High Accuracy**: Trained on comprehensive VSL dataset with data augmentation
- **Web Interface**: User-friendly Streamlit application
- **Top-3 Predictions**: Provides confidence scores for multiple predictions

## Dataset

The dataset is sourced from the Vietnamese Sign Language Dictionary by QIP EDC MOET (https://qipedc.moet.gov.vn). It contains thousands of sign language videos across various categories.

### Data Structure
```
dataset/
├── videos/          # Raw video files
├── augmented/       # Processed keypoints with augmentations
└── label/
    ├── label.csv    # Video-label mappings
    └── label_map.json  # Encoded label mappings
```

## Model Architecture

- **Input**: 60-frame sequences with 201 features (pose + hands keypoints)
- **Layers**: Bidirectional LSTM with attention mechanisms
- **Output**: Softmax classification over 3315+ sign classes

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vsl-lstm-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Download the dataset:
```bash
python download_dataset.py
```

2. Generate augmented data:
```bash
python run_augmentation.py
```

3. Train the model:
```bash
jupyter notebook training.ipynb
# Run all cells in the notebook
```

### Running the Web App

```bash
streamlit run app.py
```

The app will open in your browser with options to:
- Upload a video file
- Use webcam for real-time recognition
- View processed videos with landmark overlays

## Project Structure

```
vsl-lstm-project/
├── app.py                    # Streamlit web application
├── training.ipynb           # Model training notebook
├── run_augmentation.py      # Data augmentation script
├── download_dataset.py      # Dataset download script
├── requirements.txt         # Python dependencies
├── models/                  # Trained models
│   ├── best_model.keras
│   └── final_model.keras
├── dataset/                 # Dataset files
│   ├── videos/
│   ├── augmented/
│   └── label/
├── augmentation/            # Augmentation utilities
│   ├── __init__.py
│   ├── augmentations.py
│   ├── constants.py
│   ├── ik_solver.py
│   └── utils.py
└── README.md
```

## Key Components

### Data Augmentation (`augmentation/`)
- Scaling, rotation, translation
- Time stretching
- Inter-hand distance adjustments
- Noise injection

### Model Training (`training.ipynb`)
- Mixed precision training
- Bidirectional LSTM architecture
- Cosine decay learning rate
- Early stopping and model checkpointing

### Web Application (`app.py`)
- Real-time video processing
- Landmark visualization
- Prediction with confidence scores

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- Mediapipe
- OpenCV
- Streamlit
- CUDA-compatible GPU (recommended for training)

## Model Performance

- **Test Accuracy**: ~85% (varies by dataset)
- **Top-5 Accuracy**: ~95%
- **Input Resolution**: 60 frames @ 201 features each

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is for educational and research purposes. Please respect the original dataset licensing terms.

## Acknowledgments

- Dataset provided by QIP EDC MOET
- Mediapipe for pose estimation
- TensorFlow/Keras for deep learning framework
- Streamlit for web application framework