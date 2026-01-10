# Compatibility Analysis: `preprocess_videos.py` ↔ `extract_embeddings.py`

## Summary

✅ **COMPATIBLE** - The files are now fully compatible. `extract_embeddings.py` has been modified to read preprocessed frames directly from the directory structure created by `preprocess_videos.py`.

## Key Changes Made to `extract_embeddings.py`

### 1. **Removed Video Processing Dependencies**
- ❌ Removed `decord` library (`VideoReader`)
- ✅ Added `pathlib.Path` for file handling

### 2. **Updated Configuration**
```python
# Before:
VIDEO_ROOT = "videos/train"
NUM_FRAMES = 32

# After:
FRAMES_ROOT = "preprocessed_frames"
IMAGE_FORMAT = "png"  # Must match preprocess_videos.py config
```

### 3. **New Frame Loading Function**
Replaced video reading with directory-based frame loading:
```python
def load_frames_from_directory(frame_dir, image_format="png"):
    """Load all frames from a directory created by preprocess_videos.py"""
    frame_files = sorted(Path(frame_dir).glob(f"frame_*.{image_format}"))
    frames = [Image.open(f).convert("RGB") for f in frame_files]
    return frames
```

### 4. **Removed Redundant Frame Sampling**
- `preprocess_videos.py` already handles frame sampling (uniform/random, 32 frames by default)
- Removed `sample_frames()` function from `extract_embeddings.py`
- Now processes **all frames** in each directory

### 5. **Updated Directory Structure Processing**
```python
# Before: Loop through video files
for video_file in os.listdir(folder):
    if video_file.endswith((".mp4", ".avi", ".mov")):
        ...

# After: Loop through frame directories
for frame_dir_name in os.listdir(label_folder):
    if os.path.isdir(frame_dir_path):
        ...
```

## Configuration Compatibility

### `preprocess_videos.py` Settings → `extract_embeddings.py` Requirements

| **Aspect** | **preprocess_videos.py** | **extract_embeddings.py** | **Compatible?** |
|------------|--------------------------|---------------------------|-----------------|
| **Output Format** | `output_format = "frames"` | Expects individual frame images | ✅ Yes |
| **Image Format** | `image_format = "png"` | `IMAGE_FORMAT = "png"` | ✅ Yes |
| **Frame Count** | `num_sampled_frames = 32` | Processes all frames in directory | ✅ Yes |
| **Resolution** | `frame_resolution = 256` | DINOv3 processor resizes to 224×224 | ✅ Yes (auto-resized) |
| **Color Space** | `pixel_format = "rgb24"` | PIL opens as RGB | ✅ Yes |
| **Directory Structure** | `preprocessed_frames/{label}/{label_N}/frame_XXXX.png` | Expects same structure | ✅ Yes |

## Important Configuration Notes

### ⚠️ **Must Match Between Files**

1. **Image Format**
   - `preprocess_videos.py`: `image_format = "png"` (default)
   - `extract_embeddings.py`: `IMAGE_FORMAT = "png"`
   - If you change to `"jpg"` in preprocessing, update `IMAGE_FORMAT` accordingly

2. **Root Directory**
   - `preprocess_videos.py`: `output_root = "preprocessed_frames"`
   - `extract_embeddings.py`: `FRAMES_ROOT = "preprocessed_frames"`
   - Must point to the same directory

### ✅ **Auto-Handled Differences**

1. **Frame Resolution**
   - Preprocessing uses 256×256 (with aspect-ratio preservation)
   - DINOv3 processor automatically resizes to 224×224
   - No manual configuration needed

2. **Color Normalization**
   - Preprocessing uses RGB color space
   - DINOv3 processor applies ImageNet normalization automatically
   - No manual configuration needed

3. **Frame Count**
   - Preprocessing samples 32 frames (configurable)
   - Embedding extraction processes all frames in directory
   - Works with any number of frames

## Expected Workflow

```bash
# Step 1: Preprocess videos to extract frames
python preprocess_videos.py --output_format frames --image_format png

# Output structure:
# preprocessed_frames/
#   ├── real/
#   │   ├── real_1/
#   │   │   ├── frame_0000.png
#   │   │   ├── frame_0001.png
#   │   │   └── ... (32 frames)
#   │   ├── real_2/
#   │   └── ...
#   └── generated/
#       ├── generated_1/
#       └── ...

# Step 2: Extract embeddings from preprocessed frames
python extract_embeddings.py

# Output structure:
# embeddings/
#   ├── real_1.npy
#   ├── real_2.npy
#   ├── generated_1.npy
#   └── ...
```

## Benefits of This Approach

1. ✅ **No Double Processing** - Frames are preprocessed once, embeddings extracted from saved frames
2. ✅ **Quality Preservation** - PNG format ensures lossless storage of preprocessed frames
3. ✅ **Consistency** - All frames go through identical preprocessing pipeline
4. ✅ **Efficiency** - No need to re-decode videos for embedding extraction
5. ✅ **Flexibility** - Can easily verify or inspect preprocessed frames before embedding extraction

## Migration Guide

If you have existing video files and want to use the new pipeline:

```bash
# 1. Preprocess your videos (adjust paths as needed)
python preprocess_videos.py \
  --input_root "../real,../generated" \
  --output_root "preprocessed_frames" \
  --output_format frames \
  --image_format png \
  --num_sampled_frames 32

# 2. Extract embeddings
python extract_embeddings.py

# 3. Verify output
# Check that embeddings/ contains .npy files matching your preprocessed frame directories
```

## Troubleshooting

### Issue: "No frames found in directory"
- **Cause**: `IMAGE_FORMAT` doesn't match the actual frame format
- **Solution**: Check your `preprocess_videos.py` config and update `IMAGE_FORMAT` in `extract_embeddings.py`

### Issue: "Folder not found: preprocessed_frames/real"
- **Cause**: Preprocessing hasn't been run or output directory differs
- **Solution**: Run `preprocess_videos.py` first, or update `FRAMES_ROOT` to match your preprocessing output

### Issue: Embedding dimensions don't match expected
- **Cause**: Different number of frames than expected
- **Solution**: This is normal - embeddings shape will be (N, 384) where N is the number of frames in each directory
