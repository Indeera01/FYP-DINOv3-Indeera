import os
import sys
import subprocess
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
from tqdm import tqdm

# -------------------------------
# Configuration
# -------------------------------
DEMO_FOLDER = "DEMO"
PREPROCESSED_FOLDER = "DEMO/preprocessed_frames-demo"
EMBEDDINGS_FOLDER = "DEMO/embeddings-demo"
MODEL_PATH = "temporal_cnn_dino.pth"
NUM_FRAMES = 32
EMB_DIM = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Temporal CNN Model (same as training)
# -------------------------------
class TemporalCNN(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, num_classes=2):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv1d(emb_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, T, D) → (B, D, T)
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = x.squeeze(-1)
        return self.classifier(x)


# -------------------------------
# Main Pipeline
# -------------------------------
def main():
    print("=" * 60)
    print("🎬 DEMO Video Prediction Pipeline")
    print("=" * 60)
    print(f"📁 Demo folder: {DEMO_FOLDER}")
    print(f"🖥️  Device: {DEVICE}")
    print(f"📊 Model: {MODEL_PATH}")
    print("=" * 60)
    print()

    # Step 0: Check if DEMO folder exists and has videos
    if not os.path.exists(DEMO_FOLDER):
        print(f"❌ Error: DEMO folder '{DEMO_FOLDER}' not found!")
        print(f"   Please create the folder and add video files to it.")
        sys.exit(1)
    
    # Check for video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = set()  # Use set to avoid duplicates
    for ext in video_extensions:
        video_files.update(Path(DEMO_FOLDER).glob(f"*{ext}"))
        video_files.update(Path(DEMO_FOLDER).glob(f"*{ext.upper()}"))
    video_files = list(video_files)  # Convert back to list
    
    if len(video_files) == 0:
        print(f"❌ Error: No video files found in '{DEMO_FOLDER}' folder!")
        print(f"   Supported formats: {', '.join(video_extensions)}")
        print(f"   Please add video files to the DEMO folder and run again.")
        sys.exit(1)
    
    print(f"✅ Found {len(video_files)} video(s) in DEMO folder")
    print()

    # Step 1: Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        print("   Please train the model using train_temporal_cnn.py first.")
        sys.exit(1)
    
    print(f"✅ Model file found: {MODEL_PATH}")
    print()

    # Step 2: Preprocess videos from DEMO folder (direct processing)
    print("🔄 Step 1/3: Preprocessing videos...")
    print("-" * 60)
    
    # Create output directory
    os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)
    
    # Preprocessing configuration (match training settings)
    frame_rate = 25
    duration = 5
    resolution = 224
    resize_method = "bicubic"
    image_format = "png"
    
    processed_count = 0
    for video_idx, video_path in enumerate(tqdm(sorted(video_files), desc="  Preprocessing"), start=1):
        video_name = video_path.stem
        output_dir = os.path.join(PREPROCESSED_FOLDER, f"video_{video_idx}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract frames directly using FFmpeg
        # Sample uniformly across the video
        total_frames = duration * frame_rate
        
        for frame_idx in range(NUM_FRAMES):
            # Calculate timestamp for uniform sampling
            timestamp = (frame_idx / NUM_FRAMES) * duration
            output_frame = os.path.join(output_dir, f"frame_{frame_idx:04d}.{image_format}")
            
            # Build FFmpeg command for single frame extraction with preprocessing
            # Use force_original_aspect_ratio=increase to handle both portrait and landscape
            # This scales so the smaller dimension becomes 224, then crops to 224x224 (center)
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(timestamp),
                "-i", str(video_path),
                "-vf", f"fps={frame_rate},scale={resolution}:{resolution}:force_original_aspect_ratio=increase:flags={resize_method},crop={resolution}:{resolution}",
                "-vframes", "1",
                "-pix_fmt", "rgb24",
                output_frame
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                pass  # Skip failed frames
        
        # Verify we extracted frames
        extracted = len([f for f in os.listdir(output_dir) if f.endswith(f".{image_format}")])
        if extracted > 0:
            processed_count += 1
    
    print(f"\n✅ Preprocessing complete! Processed {processed_count}/{len(video_files)} videos")
    print()

    # Step 3: Extract embeddings using a temporary script
    print("🔄 Step 2/3: Extracting embeddings...")
    print("-" * 60)
    
    # Create temporary extraction script
    extract_script = f"""
import os
import sys
import numpy as np
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from pathlib import Path
from tqdm import tqdm

FRAMES_ROOT = "{PREPROCESSED_FOLDER}"
OUTPUT_FOLDER = "{EMBEDDINGS_FOLDER}"
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGE_FORMAT = "png"
DEVICE = "{DEVICE}"

print("Loading DINOv3 model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("✅ Model loaded successfully!")
print()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_embeddings_from_frames(frame_dir, image_format="png"):
    frame_files = sorted(Path(frame_dir).glob(f"frame_*.{{image_format}}"))
    if len(frame_files) == 0:
        raise ValueError(f"No frames found in {{frame_dir}}")
    
    embeddings = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
        
        embeddings.append(cls_emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    return embeddings

# Process all video directories
if not os.path.exists(FRAMES_ROOT):
    print(f"Error: Preprocessed frames folder not found: {{FRAMES_ROOT}}")
    sys.exit(1)

video_dirs = [d for d in os.listdir(FRAMES_ROOT) if os.path.isdir(os.path.join(FRAMES_ROOT, d))]
print(f"Found {{len(video_dirs)}} video(s) to process")
print()

if len(video_dirs) == 0:
    print("Warning: No video directories found in preprocessed frames folder")
    sys.exit(1)

for video_dir in tqdm(video_dirs, desc="Extracting embeddings"):
    frame_dir_path = os.path.join(FRAMES_ROOT, video_dir)
    try:
        emb = extract_embeddings_from_frames(frame_dir_path, IMAGE_FORMAT)
        save_name = f"{{video_dir}}.npy"
        np.save(os.path.join(OUTPUT_FOLDER, save_name), emb)
    except Exception as e:
        print(f"\\nWarning: Skipping {{video_dir}}: {{e}}")

print()
print("✅ Embedding extraction complete!")
"""
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        temp_script_path = f.name
        f.write(extract_script)
    
    try:
        # Use same UTF-8 environment for embedding extraction
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(["python", temp_script_path], capture_output=True, text=True, env=env)
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Error during embedding extraction:")
            print(result.stderr)
            sys.exit(1)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
    
    print()

    # Step 4: Load model and make predictions
    print("🔄 Step 3/3: Making predictions...")
    print("-" * 60)
    
    # Load the trained model
    model = TemporalCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded from {MODEL_PATH}")
    print()

    # Get all embedding files
    if not os.path.exists(EMBEDDINGS_FOLDER):
        print(f"❌ Embeddings folder not found: {EMBEDDINGS_FOLDER}")
        sys.exit(1)
    
    embedding_files = sorted([f for f in os.listdir(EMBEDDINGS_FOLDER) if f.endswith(".npy")])
    
    if len(embedding_files) == 0:
        print(f"❌ No embeddings found in {EMBEDDINGS_FOLDER}")
        sys.exit(1)

    print(f"📊 Found {len(embedding_files)} video(s) to classify")
    print("=" * 60)
    print()

    # Make predictions
    for emb_file in embedding_files:
        emb_path = os.path.join(EMBEDDINGS_FOLDER, emb_file)
        video_name = emb_file.replace(".npy", "")
        
        try:
            # Load embeddings
            embeddings = np.load(emb_path)
            
            # Validate shape
            if embeddings.shape != (NUM_FRAMES, EMB_DIM):
                print(f"⚠️  Warning: Unexpected embedding shape for {video_name}: {embeddings.shape}")
                print(f"   Expected: ({NUM_FRAMES}, {EMB_DIM})")
                print(f"   Skipping this video...")
                print()
                continue
            
            # Convert to tensor and add batch dimension
            x = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Make prediction
            with torch.no_grad():
                output = model(x)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(output, dim=1).item()
                confidence = probabilities[0, prediction].item()
            
            # Get probabilities for both classes
            prob_real = probabilities[0, 0].item()
            prob_fake = probabilities[0, 1].item()
            
            # Determine label
            label = "GENERATED/DEEPFAKE" if prediction == 1 else "REAL"
            
            # Print results with nice formatting
            print(f"📹 Video: {video_name}")
            print(f"   🎯 Prediction: {label}")
            print(f"   💯 Confidence: {confidence * 100:.2f}%")
            print(f"   📊 Probabilities:")
            print(f"      • Real: {prob_real * 100:.2f}%")
            print(f"      • Generated/Deepfake: {prob_fake * 100:.2f}%")
            print()
            
        except Exception as e:
            print(f"❌ Error processing {video_name}: {e}")
            print()

    print("=" * 60)
    print("🎉 Prediction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
