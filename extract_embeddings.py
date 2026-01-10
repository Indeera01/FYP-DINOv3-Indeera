import os
import numpy as np
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# -------------------------------
# Configuration
# -------------------------------
DATASET_SPLITS = ["train", "validation", "test"]  # Dataset splits to process
FRAMES_ROOT_TEMPLATE = "preprocessed_frames-{split}"  # Use {split} placeholder
OUTPUT_FOLDER_TEMPLATE = "embeddings-{split}"  # Use {split} placeholder
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGE_FORMAT = "png"  # Must match the format in preprocess_videos.py config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("🧠 DINOv3 Embedding Extraction")
print("=" * 60)
print(f"Dataset splits: {', '.join(DATASET_SPLITS)}")
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Image format: {IMAGE_FORMAT}")
print("=" * 60)
print()

# -------------------------------
# Load model
# -------------------------------
print("Loading DINOv3 model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


# -------------------------------
# Load frames from directory
# -------------------------------
def load_frames_from_directory(frame_dir, image_format="png"):
    """
    Load all frames from a directory created by preprocess_videos.py.
    
    Args:
        frame_dir: Path to directory containing frame images
        image_format: Image format (png or jpg)
    
    Returns:
        List of PIL Images sorted by frame number
    """
    frame_files = sorted(Path(frame_dir).glob(f"frame_*.{image_format}"))
    
    if len(frame_files) == 0:
        raise ValueError(f"No frames found in {frame_dir}")
    
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert("RGB")
        frames.append(img)
    
    return frames


# -------------------------------
# Embedding extraction
# -------------------------------
def extract_embeddings_from_frames(frame_dir, image_format="png"):
    """
    Extract DINOv3 embeddings from preprocessed frames.
    
    Args:
        frame_dir: Directory containing preprocessed frames
        image_format: Image format (png or jpg)
    
    Returns:
        numpy array of shape (num_frames, embedding_dim)
    """
    frames = load_frames_from_directory(frame_dir, image_format)
    
    embeddings = []
    
    for frame in frames:
        inputs = processor(images=frame, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, embedding_dim)
        
        embeddings.append(cls_emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings)  # (num_frames, embedding_dim)
    return embeddings


# -------------------------------
# Main loop
# -------------------------------
total_extracted_all = 0
total_skipped_all = 0

for split in DATASET_SPLITS:
    print(f"\n{'=' * 60}")
    print(f"📦 Processing '{split.upper()}' split")
    print("=" * 60)
    
    # Format paths for this split
    frames_root = FRAMES_ROOT_TEMPLATE.format(split=split)
    output_folder = OUTPUT_FOLDER_TEMPLATE.format(split=split)
    
    # Create output folder for this split
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(frames_root):
        print(f"⚠️  Warning: Frames directory not found: {frames_root}")
        print(f"   Skipping {split} split...")
        continue
    
    # Track stats for this split
    total_extracted = 0
    total_skipped = 0
    
    for label in ["real", "generated"]:
        label_folder = os.path.join(frames_root, label)
        
        if not os.path.exists(label_folder):
            print(f"⚠️  Warning: Folder not found: {label_folder}")
            continue
        
        # Get all frame directories (e.g., real_1, real_2, etc.)
        frame_dirs = [d for d in os.listdir(label_folder) 
                      if os.path.isdir(os.path.join(label_folder, d))]
        
        print(f"📁 Found {len(frame_dirs)} {label} videos")
        
        for frame_dir_name in tqdm(frame_dirs, desc=f"  {split}/{label}"):
            frame_dir_path = os.path.join(label_folder, frame_dir_name)
            
            try:
                emb = extract_embeddings_from_frames(frame_dir_path, IMAGE_FORMAT)
                
                # Save with consistent naming: frame_dir_name already contains label
                save_name = f"{frame_dir_name}.npy"
                np.save(os.path.join(output_folder, save_name), emb)
                total_extracted += 1
                
            except ValueError as e:
                print(f"\n⚠️  Skipping {frame_dir_name}: {e}")
                total_skipped += 1
                continue
    
    # Summary for this split
    print(f"\n✅ {split.upper()} - Extracted: {total_extracted} videos, Skipped: {total_skipped}")
    print(f"📂 Output: {output_folder}")
    
    # Accumulate totals
    total_extracted_all += total_extracted
    total_skipped_all += total_skipped

# Final summary across all splits
print()
print("=" * 60)
print("🎉 ALL SPLITS EMBEDDING EXTRACTION COMPLETE")
print("=" * 60)
print(f"✅ Total embeddings extracted: {total_extracted_all}")
if total_skipped_all > 0:
    print(f"⚠️  Total skipped: {total_skipped_all}")
print("=" * 60)


