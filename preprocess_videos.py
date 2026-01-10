import os
import subprocess
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm

# -------------------------------
# Default Configuration
# -------------------------------
DEFAULT_CONFIG = {
    # Temporal Parameters
    "frame_rate": 25,                    # FPS: 15, 25
    "duration": 5,                       # seconds: 5, 10 (videos shorter will be skipped)
    "frame_sampling_strategy": "uniform", # uniform, random
    "num_sampled_frames": 32,            # 16, 32
    
    # Spatial Parameters
    "frame_resolution": 224,             # 256, 384
    "resize_method": "bicubic",         # bilinear, bicubic
    
    # Color & Format Parameters
    "color_space": "RGB",                # RGB, YUV
    "pixel_format": "rgb24",             # rgb24
    "color_normalization": "imagenet",   # ImageNet stats (applied during training)
    
    # Encoding & Data Integrity Parameters
    "video_codec": "libx264",            # H.264
    "bitrate": "2M",                     # 1-3 Mbps (e.g., "1M", "2M", "3M")
    "audio": "removed",                  # removed, kept
    "container_format": "mp4",           # MP4
    
    # I/O Parameters
    "dataset_splits": ["train", "validation", "test"],  # Dataset splits to process
    "input_root_template": "../datasets/{split}/real,../datasets/{split}/generated",  # Use {split} placeholder
    "output_root_template": "preprocessed_frames-{split}",  # Use {split} placeholder
    "video_extensions": ".mp4,.avi,.mov,.mkv",
    
    # Processing Parameters
    "skip_duration_check": False,         # Skip duration check if ffprobe unavailable
    "output_format": "frames",            # "video" or "frames" (frames is more efficient for feature extraction)
    "image_format": "png"                 # "png" (lossless, best for research) or "jpg" (when output_format="frames")
}

# ImageNet normalization stats (for reference, applied during training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -------------------------------
# Video Preprocessing Function
# -------------------------------
def get_video_duration(video_path):
    """
    Get the duration of a video in seconds using ffprobe.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Duration in seconds or None if error
    """
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except:
        return None


def preprocess_video(input_path, output_path, config):
    """
    Preprocess a video using FFmpeg with configurable parameters.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        config: Dictionary containing preprocessing parameters
    
    Returns:
        True if successful, False otherwise, "skip" if video too short
    """
    # Check video duration
    video_duration = get_video_duration(input_path)
    if video_duration is None:
        print(f"⚠️  Could not determine duration for {input_path}")
        return False
    
    if video_duration < config["duration"]:
        return "skip"
    
    # Build FFmpeg command
    cmd = ["ffmpeg", "-y", "-i", input_path]
    
    # Duration limit (trim to exact duration)
    cmd.extend(["-t", str(config["duration"])])
    
    # Build video filter chain
    vf_filters = []
    
    # Frame rate
    vf_filters.append(f"fps={config['frame_rate']}")
    
    # Resize with specified method
    resize_method = config["resize_method"]
    if resize_method == "bilinear":
        flags = "bilinear"
    elif resize_method == "bicubic":
        flags = "bicubic"
    else:
        flags = "bilinear"
    
    resolution = config["frame_resolution"]
    # Aspect-preserving scale + center padding to avoid distortion
    vf_filters.append(f"scale={resolution}:-1:flags={flags},pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2")
    
    # Color space conversion (if needed)
    if config["color_space"].upper() == "YUV":
        vf_filters.append("format=yuv420p")
    
    # Combine all video filters
    cmd.extend(["-vf", ",".join(vf_filters)])
    
    # Pixel format
    cmd.extend(["-pix_fmt", config["pixel_format"]])
    
    # Video codec
    cmd.extend(["-c:v", config["video_codec"]])
    
    # Bitrate
    cmd.extend(["-b:v", config["bitrate"]])
    
    # Audio handling
    if config["audio"] == "removed":
        cmd.append("-an")
    else:
        cmd.extend(["-c:a", "copy"])
    
    # Output file
    cmd.append(output_path)
    
    # Execute FFmpeg
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_path}: {e.stderr.decode()}")
        return False


def sample_frames_from_video(video_path, config):
    """
    Sample frames from a video based on the sampling strategy.
    This function extracts individual frames as images.
    
    Args:
        video_path: Path to the video file
        config: Dictionary containing preprocessing parameters
    
    Returns:
        List of frame paths or None if error
    """
    output_dir = Path(video_path).parent / f"{Path(video_path).stem}_frames"
    output_dir.mkdir(exist_ok=True)
    
    num_frames = config["num_sampled_frames"]
    strategy = config["frame_sampling_strategy"]
    
    # Get total number of frames in video
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        total_frames = int(result.stdout.strip())
    except:
        # Fallback: estimate from duration and fps
        total_frames = config["duration"] * config["frame_rate"]
    
    # Determine which frames to extract
    if strategy == "uniform":
        # Uniform sampling
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]
    elif strategy == "random":
        # Random sampling
        max_frames = min(total_frames, total_frames)
        frame_indices = sorted(random.sample(range(max_frames), min(num_frames, max_frames)))
    else:
        frame_indices = list(range(min(num_frames, total_frames)))
    
    return frame_indices, output_dir


def extract_frames_to_images(input_path, output_dir, config):
    """
    Extract frames from video and save as images.
    
    Args:
        input_path: Path to input video
        output_dir: Directory to save frames
        config: Dictionary containing preprocessing parameters
    
    Returns:
        True if successful, False otherwise, "skip" if video too short
    """
    # Check video duration (if not skipped)
    if not config.get("skip_duration_check", False):
        video_duration = get_video_duration(input_path)
        if video_duration is None:
            print(f"⚠️  Could not determine duration for {input_path}")
            return False
        
        if video_duration < config["duration"]:
            return "skip"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine frame sampling
    num_frames = config["num_sampled_frames"]
    strategy = config["frame_sampling_strategy"]
    image_format = config["image_format"]
    resolution = config["frame_resolution"]
    resize_method = config["resize_method"]
    
    # Build resize/scale filter for aspect-preserving transformation
    if resize_method == "bilinear":
        flags = "bilinear"
    elif resize_method == "bicubic":
        flags = "bicubic"
    else:
        flags = "bilinear"
    
    # Get total frames directly from source video (after trimming and FPS adjustment)
    # This avoids creating an intermediate temp video
    try:
        # Calculate expected frames after trimming and FPS adjustment
        total_frames = config["duration"] * config["frame_rate"]
    except:
        total_frames = config["duration"] * config["frame_rate"]
    
    # Determine frame indices to sample
    if strategy == "uniform":
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            frame_indices = [int(i * step) for i in range(num_frames)]
    elif strategy == "random":
        max_frames = min(total_frames, total_frames)
        frame_indices = sorted(random.sample(range(max_frames), min(num_frames, max_frames)))
    else:
        frame_indices = list(range(min(num_frames, total_frames)))
    
    # Extract frames directly from source video (no intermediate re-encoding)
    # This preserves quality and avoids lossy compression artifacts
    for idx, frame_num in enumerate(frame_indices):
        output_frame = os.path.join(output_dir, f"frame_{idx:04d}.{image_format}")
        
        # Calculate timestamp for this frame
        timestamp = frame_num / config["frame_rate"]
        
        # Build filter chain: trim → fps → scale+pad
        vf_filters = [
            f"fps={config['frame_rate']}",
            f"scale={resolution}:-1:flags={flags},pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2"
        ]
        
        # Extract single frame directly with all preprocessing in one pass
        cmd_extract = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),  # Seek to specific timestamp
            "-i", input_path,
            "-t", str(config["duration"]),  # Respect duration limit
            "-vf", ",".join(vf_filters),
            "-vframes", "1",  # Extract exactly 1 frame
            "-pix_fmt", config["pixel_format"],
            output_frame
        ]
        
        try:
            subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError:
            pass  # Skip failed frame extractions
    
    # Verify we extracted frames
    extracted_frames = len([f for f in os.listdir(output_dir) if f.endswith(f".{image_format}")])
    if extracted_frames > 0:
        return True
    else:
        return False


# -------------------------------
# Main Processing Loop
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Video Preprocessing for Deepfake Detection")
    
    # Config file option
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Temporal parameters
    parser.add_argument("--frame_rate", type=int, help="Target frame rate (FPS)")
    parser.add_argument("--duration", type=int, help="Target duration in seconds (skip videos shorter than this)")
    parser.add_argument("--frame_sampling_strategy", choices=["uniform", "random"], 
                        help="Frame sampling strategy")
    parser.add_argument("--num_sampled_frames", type=int, help="Number of frames to sample")
    
    # Spatial parameters
    parser.add_argument("--frame_resolution", type=int, help="Target frame resolution (square)")
    parser.add_argument("--resize_method", choices=["bilinear", "bicubic"], 
                        help="Resize interpolation method")
    
    # Color & format parameters
    parser.add_argument("--color_space", choices=["RGB", "YUV"], help="Color space")
    parser.add_argument("--pixel_format", type=str, help="Pixel format")
    
    # Encoding parameters
    parser.add_argument("--video_codec", type=str, help="Video codec")
    parser.add_argument("--bitrate", type=str, help="Video bitrate (e.g., '2M')")
    parser.add_argument("--audio", choices=["removed", "kept"], help="Audio handling")
    
    # I/O parameters
    parser.add_argument("--dataset_splits", type=str,
                        help="Comma-separated dataset splits to process (e.g., 'train,validation,test')")
    parser.add_argument("--input_root_template", type=str, 
                        help="Input directory template with {split} placeholder (e.g., '../temp-datasets/{split}/real,../temp-datasets/{split}/generated')")
    parser.add_argument("--output_root_template", type=str, 
                        help="Output directory template with {split} placeholder (e.g., 'preprocessed_frames-{split}')")
    parser.add_argument("--video_extensions", type=str, 
                        help="Comma-separated video extensions")
    
    # Processing parameters
    parser.add_argument("--output_format", choices=["video", "frames"],
                        help="Output format: 'video' or 'frames' (frames is more efficient for feature extraction)")
    parser.add_argument("--image_format", choices=["jpg", "png"],
                        help="Image format when output_format='frames'")
    parser.add_argument("--skip_duration_check", action="store_true",
                        help="Skip duration check (use when ffprobe unavailable)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    # Load from config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Override with command-line arguments
    for key, value in vars(args).items():
        if value is not None and key != "config":
            # Special handling for dataset_splits
            if key == "dataset_splits" and isinstance(value, str):
                config[key] = [s.strip() for s in value.split(",")]
            else:
                config[key] = value
    
    # Get dataset splits and templates
    dataset_splits = config["dataset_splits"]
    input_template = config["input_root_template"]
    output_template = config["output_root_template"]
    video_exts = tuple(config["video_extensions"].split(","))
    
    # Print configuration
    print("=" * 60)
    print("VIDEO PREPROCESSING CONFIGURATION")
    print("=" * 60)
    print(f"Dataset Splits: {', '.join(dataset_splits)}")
    print(f"Output Format: {config['output_format'].upper()}")
    print(f"Frame Rate (FPS): {config['frame_rate']}")
    print(f"Target Duration: {config['duration']}s (skip videos shorter than this)")
    print(f"Frame Sampling: {config['frame_sampling_strategy']} ({config['num_sampled_frames']} frames)")
    print(f"Resolution: {config['frame_resolution']}x{config['frame_resolution']}")
    print(f"Resize Method: {config['resize_method']}")
    print(f"Color Space: {config['color_space']}")
    print(f"Pixel Format: {config['pixel_format']}")
    if config['output_format'] == 'video':
        print(f"Video Codec: {config['video_codec']}")
        print(f"Bitrate: {config['bitrate']}")
        print(f"Audio: {config['audio']}")
        print(f"Container Format: {config['container_format']}")
    else:
        print(f"Image Format: {config['image_format'].upper()}")
    print("=" * 60)
    print()
    
    # Process each dataset split
    total_processed_all = 0
    total_failed_all = 0
    total_skipped_all = 0
    
    for split in dataset_splits:
        print(f"\n{'=' * 60}")
        print(f"📦 Processing '{split.upper()}' split")
        print("=" * 60)
        
        # Format paths for this split
        input_root = input_template.format(split=split)
        output_root = output_template.format(split=split)
        
        input_dirs = [d.strip() for d in input_root.split(",")]
        os.makedirs(output_root, exist_ok=True)
        
        # Track stats for this split
        total_processed = 0
        total_failed = 0
        total_skipped = 0
        
        for input_dir in input_dirs:
            if not os.path.exists(input_dir):
                print(f"⚠️  Warning: Input directory not found: {input_dir}")
                continue
            
            # Determine label from directory name
            label = os.path.basename(input_dir.rstrip("/\\"))
            output_dir = os.path.join(output_root, label)
            os.makedirs(output_dir, exist_ok=True)
            
            # Find all video files
            video_files = []
            for ext in video_exts:
                video_files.extend(Path(input_dir).glob(f"*{ext}"))
                video_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
            
            video_files = list(set(video_files))  # Remove duplicates
            
            print(f"📁 Processing '{label}': {len(video_files)} videos found")
            
            # Process each video with sequential numbering
            video_counter = 1
            for video_path in tqdm(video_files, desc=f"  {split}/{label}"):
                # Choose processing mode
                if config['output_format'] == 'frames':
                    # Extract frames to images
                    frame_output_dir = os.path.join(output_dir, f"{label}_{video_counter}")
                    result = extract_frames_to_images(str(video_path), frame_output_dir, config)
                else:
                    # Encode as video
                    output_filename = f"{label}_{video_counter}.{config['container_format']}"
                    output_path = os.path.join(output_dir, output_filename)
                    result = preprocess_video(str(video_path), output_path, config)
                
                if result == "skip":
                    total_skipped += 1
                elif result:
                    total_processed += 1
                    video_counter += 1  # Only increment counter for successful videos
                else:
                    total_failed += 1
        
        # Summary for this split
        print(f"\n✅ {split.upper()} - Processed: {total_processed}, Skipped: {total_skipped}, Failed: {total_failed}")
        print(f"📂 Output: {output_root}")
        
        # Accumulate totals
        total_processed_all += total_processed
        total_failed_all += total_failed
        total_skipped_all += total_skipped
    
    # Final summary across all splits
    print()
    print("=" * 60)
    print("🎉 ALL SPLITS PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"✅ Total successfully processed: {total_processed_all} videos")
    if total_skipped_all > 0:
        print(f"⏭️  Total skipped (too short): {total_skipped_all} videos")
    if total_failed_all > 0:
        print(f"❌ Total failed: {total_failed_all} videos")
    print("=" * 60)


if __name__ == "__main__":
    main()
