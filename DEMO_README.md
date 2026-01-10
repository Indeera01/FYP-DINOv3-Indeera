# Demo Prediction Pipeline

This script (`predict_demo.py`) provides an end-to-end pipeline for preprocessing videos, extracting embeddings, and making predictions using the trained deepfake detection model.

## Prerequisites

1. **Trained Model**: Ensure you have trained the model using `train_temporal_cnn.py`. The model file `temporal_cnn_dino.pth` should exist in the project directory.

2. **Video Files**: Place your test videos in the `DEMO` folder. Supported formats:
   - `.mp4`
   - `.avi`
   - `.mov`
   - `.mkv`

## Usage

Simply run the script:

```bash
python predict_demo.py
```

## What It Does

The script automatically performs three steps:

### Step 1: Preprocessing Videos
- Reads videos from the `DEMO` folder
- Extracts frames using the same preprocessing pipeline as training (`preprocess_videos.py`)
- Saves preprocessed frames to `preprocessed_frames-demo/`
- Configuration:
  - Frame rate: 25 FPS
  - Duration: 5 seconds
  - Number of frames: 32
  - Resolution: 224x224
  - Format: PNG (lossless)

### Step 2: Extracting Embeddings
- Uses DINOv3 model to extract embeddings from preprocessed frames
- Processes each frame through the pre-trained vision transformer
- Saves embeddings to `embeddings-demo/`
- Each video results in a `.npy` file with shape (32, 384)

### Step 3: Making Predictions
- Loads the trained temporal CNN model (`temporal_cnn_dino.pth`)
- Processes embeddings through the classifier
- Outputs predictions with confidence scores

## Output Format

For each video, the script displays:

```
📹 Video: [video_name]
   🎯 Prediction: REAL / GENERATED/DEEPFAKE
   💯 Confidence: XX.XX%
   📊 Probabilities:
      • Real: XX.XX%
      • Generated/Deepfake: XX.XX%
```

## Example

```bash
# 1. Add videos to DEMO folder
# 2. Run prediction script
python predict_demo.py

# Expected output:
# 🎬 DEMO Video Prediction Pipeline
# ============================================================
# 📁 Demo folder: DEMO
# 🖥️  Device: cuda
# 📊 Model: temporal_cnn_dino.pth
# ============================================================
# 
# ✅ Found 3 video(s) in DEMO folder
# 
# 🔄 Step 1/3: Preprocessing videos...
# ...
# ✅ Preprocessing complete!
# 
# 🔄 Step 2/3: Extracting embeddings...
# ...
# ✅ Embedding extraction complete!
# 
# 🔄 Step 3/3: Making predictions...
# ...
# 📹 Video: video_1
#    🎯 Prediction: REAL
#    💯 Confidence: 92.45%
#    📊 Probabilities:
#       • Real: 92.45%
#       • Generated/Deepfake: 7.55%
```

## Notes

- The script will skip videos shorter than 5 seconds (or extract what's available if `--skip_duration_check` is used)
- All intermediate files are saved for inspection:
  - Preprocessed frames: `preprocessed_frames-demo/`
  - Embeddings: `embeddings-demo/`
- The script validates embedding shapes before prediction
- GPU acceleration is used automatically if available

## Troubleshooting

### Error: "DEMO folder not found"
Create the DEMO folder and add video files:
```bash
mkdir DEMO
# Copy your videos to DEMO folder
```

### Error: "No video files found"
Ensure your videos have supported extensions (.mp4, .avi, .mov, .mkv)

### Error: "Model file not found"
Train the model first:
```bash
python train_temporal_cnn.py
```

### Error: "Unexpected embedding shape"
This usually means the video didn't have enough frames. The script will skip these automatically.
