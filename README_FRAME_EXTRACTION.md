# Video Preprocessing Pipeline - Frame Extraction Mode

## Overview

The preprocessing script now supports **two output modes**:
1. **Video mode**: Encodes preprocessed videos (original behavior)
2. **Frames mode**: Extracts sampled frames as images (optimized for feature extraction)

## Why Use Frame Extraction Mode?

When using frames mode, you eliminate the overhead of:
- ❌ Video encoding during preprocessing  
- ❌ Video decoding during feature extraction

Instead, you directly:
- ✅ Extract frames → Save as images → Load images for DINOv3

This is **faster** and **saves disk space** compared to encoding/decoding videos.

## Configuration

### Default Settings (Frames Mode)
```python
{
    "output_format": "frames",        # Output mode: "video" or "frames"
    "image_format": "jpg",            # Image format: "jpg" or "png"
    "num_sampled_frames": 32,         # Number of frames to extract
    "frame_sampling_strategy": "uniform",  # Sampling: "uniform" or "random"
    "frame_resolution": 256,          # Frame resolution
    "duration": 5,                    # Target video duration (skip shorter videos)
    "frame_rate": 25                  # FPS
}
```

## Usage Examples

### 1. Extract Frames (Recommended for DINOv3)
```bash
python preprocess_videos.py
```

Output structure:
```
preprocessed_frames/
├── real/
│   ├── real_1/
│   │   ├── frame_0000.jpg
│   │   ├── frame_0001.jpg
│   │   └── ... (32 frames total)
│   ├── real_2/
│   └── ...
└── generated/
    ├── generated_1/
    ├── generated_2/
    └── ...
```

### 2. Save as Videos (Original Mode)
```bash
python preprocess_videos.py --output_format video --output_root preprocessed_videos
```

Output structure:
```
preprocessed_videos/
├── real/
│   ├── real_1.mp4
│   ├── real_2.mp4
│   └── ...
└── generated/
    ├── generated_1.mp4
    ├── generated_2.mp4
    └── ...
```

### 3. Custom Configuration
```bash
# Extract 16 frames at 384px resolution as PNG
python preprocess_videos.py \
    --num_sampled_frames 16 \
    --frame_resolution 384 \
    --image_format png \
    --frame_sampling_strategy random
```

## Updated Feature Extraction Code

With frame extraction mode, your feature extraction becomes simpler:

```python
import os
import numpy as np
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from tqdm import tqdm

# Configuration
FRAMES_ROOT = "preprocessed_frames"
OUTPUT_FOLDER = "embeddings"
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
print("Loading DINOv3 model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Extract embeddings from frames
def extract_embeddings_from_frames(frame_dir):
    """Load frames and extract embeddings"""
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
    embeddings = []
    
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = Image.open(frame_path)
        
        inputs = processor(images=frame, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, 384)
        
        embeddings.append(cls_emb.cpu().numpy())
    
    return np.vstack(embeddings)  # (32, 384)

# Process all frame directories
for label in ["real", "generated"]:
    label_dir = os.path.join(FRAMES_ROOT, label)
    frame_dirs = sorted([d for d in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, d))])
    
    for frame_dir_name in tqdm(frame_dirs, desc=f"Extracting {label}"):
        frame_dir = os.path.join(label_dir, frame_dir_name)
        emb = extract_embeddings_from_frames(frame_dir)
        
        save_name = f"{frame_dir_name}_{label}.npy"
        np.save(os.path.join(OUTPUT_FOLDER, save_name), emb)

print("✅ Embedding extraction complete.")
```

## Benefits

| Aspect | Video Mode | **Frames Mode** (Recommended) |
|--------|------------|-------------------------------|
| Preprocessing speed | 🟡 Slower (encoding) | 🟢 Faster (direct extraction) |
| Feature extraction | 🟡 Slower (decoding) | 🟢 Faster (load images) |
| Disk space | 🟡 More (compressed video) | 🟢 Less (raw frames) |
| Flexibility | ❌ Need VideoReader | ✅ Simple PIL/cv2 |
| Pipeline complexity | 🟡 More steps | 🟢 Simpler |

## Configuration File Example

Create `config.json`:
```json
{
    "output_format": "frames",
    "image_format": "jpg",
    "num_sampled_frames": 32,
    "frame_sampling_strategy": "uniform",
    "frame_resolution": 256,
    "duration": 5,
    "frame_rate": 25,
    "input_root": "../real,../generated",
    "output_root": "preprocessed_frames"
}
```

Run with config:
```bash
python preprocess_videos.py --config config.json
```
