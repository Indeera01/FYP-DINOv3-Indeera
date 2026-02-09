"""
Multi-Processing Video Preprocessing Script
============================================
This is an enhanced version of preprocess_videos.py with parallel processing support.
It processes multiple videos simultaneously using Python's multiprocessing module.

Key improvements:
- Parallel processing using multiple CPU cores
- Significant speedup for large datasets
- Progress tracking with tqdm
- Proper error handling for parallel execution
"""

import os
import subprocess
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

# -------------------------------
# Default Configuration
# -------------------------------
DEFAULT_CONFIG = {
    # Temporal Parameters
    "frame_rate": 25,                    
    "duration": 5,                       
    "frame_sampling_strategy": "uniform", 
    "num_sampled_frames": 125,            
    
    # Spatial Parameters
    "frame_resolution": 224,             
    "resize_method": "bicubic",         
    
    # Color & Format Parameters
    "color_space": "RGB",                
    "pixel_format": "rgb24",             
    "color_normalization": "imagenet",   
    
    # Encoding & Data Integrity Parameters
    "video_codec": "libx264",            
    "bitrate": "2M",                     
    "audio": "removed",                  
    "container_format": "mp4",           
    
    # I/O Parameters
    "dataset_splits": ["train", "validation", "test"],  
    "input_root_template": "../datasets/{split}/real,../datasets/{split}/generated",  
    "output_root_template": "preprocessed_frames-{split}",  
    "video_extensions": ".mp4,.avi,.mov,.mkv",
    
    # Processing Parameters
    "skip_duration_check": False,         
    "output_format": "frames",            
    "image_format": "png",                
    "num_workers": cpu_count()            # Number of parallel workers
}

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_video_duration(video_path):
    """Get the duration of a video in seconds using ffprobe."""
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


def extract_frames_to_images(input_path, output_dir, config):
    """
    Extract frames from video and save as images.
    This is designed to work with multiprocessing.
    """
    # Check video duration
    if not config.get("skip_duration_check", False):
        video_duration = get_video_duration(input_path)
        if video_duration is None:
            return ("skip", f"Could not determine duration for {input_path}")
        
        if video_duration < config["duration"]:
            return ("skip", f"Video too short: {input_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine frame sampling
    num_frames = config["num_sampled_frames"]
    strategy = config["frame_sampling_strategy"]
    image_format = config["image_format"]
    resolution = config["frame_resolution"]
    resize_method = config["resize_method"]
    
    # Build resize/scale filter
    if resize_method == "bilinear":
        flags = "bilinear"
    elif resize_method == "bicubic":
        flags = "bicubic"
    else:
        flags = "bilinear"
    
    # Calculate expected frames 
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
    
    # Extract frames
    for idx, frame_num in enumerate(frame_indices):
        output_frame = os.path.join(output_dir, f"frame_{idx:04d}.{image_format}")
        
        timestamp = frame_num / config["frame_rate"]
        
        vf_filters = [
            f"fps={config['frame_rate']}",
            f"scale={resolution}:-1:flags={flags},pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2"
        ]
        
        cmd_extract = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", input_path,
            "-t", str(config["duration"]),
            "-vf", ",".join(vf_filters),
            "-vframes", "1",
            "-pix_fmt", config["pixel_format"],
            output_frame
        ]
        
        try:
            subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError:
            pass
    
    # Verify extraction
    extracted_frames = len([f for f in os.listdir(output_dir) if f.endswith(f".{image_format}")])
    if extracted_frames > 0:
        return ("success", None)
    else:
        return ("failed", f"No frames extracted for {input_path}")


def process_single_video(args):
    """
    Process a single video - wrapper function for multiprocessing.
    Args is a tuple: (video_path, output_dir, config, label, video_counter)
    """
    video_path, output_dir, config, label, video_counter = args
    
    frame_output_dir = os.path.join(output_dir, f"{label}_{video_counter}")
    result_status, error_msg = extract_frames_to_images(str(video_path), frame_output_dir, config)
    
    return (result_status, error_msg, str(video_path))


def main():
    parser = argparse.ArgumentParser(description="Video Preprocessing for Deepfake Detection (Multi-Processing)")
    
    # Config file option
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Add all other arguments (abbreviated for brevity)
    parser.add_argument("--num_workers", type=int, help="Number of parallel workers (default: all CPU cores)")
    parser.add_argument("--dataset_splits", type=str, help="Comma-separated dataset splits")
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Override with command-line arguments
    for key, value in vars(args).items():
        if value is not None and key != "config":
            if key == "dataset_splits" and isinstance(value, str):
                config[key] = [s.strip() for s in value.split(",")]
            else:
                config[key] = value
    
    # Get configuration
    dataset_splits = config["dataset_splits"]
    input_template = config["input_root_template"]
    output_template = config["output_root_template"]
    video_exts = tuple(config["video_extensions"].split(","))
    num_workers = config.get("num_workers", cpu_count())
    
    # Print configuration
    print("=" * 60)
    print("VIDEO PREPROCESSING (MULTI-PROCESSING)")
    print("=" * 60)
    print(f"Dataset Splits: {', '.join(dataset_splits)}")
    print(f"Number of Workers: {num_workers} (CPU cores: {cpu_count()})")
    print(f"Frame Sampling: {config['frame_sampling_strategy']} ({config['num_sampled_frames']} frames)")
    print(f"Resolution: {config['frame_resolution']}x{config['frame_resolution']}")
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
        
        # Format paths
        input_root = input_template.format(split=split)
        output_root = output_template.format(split=split)
        input_dirs = [d.strip() for d in input_root.split(",")]
        os.makedirs(output_root, exist_ok=True)
        
        total_processed = 0
        total_failed = 0
        total_skipped = 0
        
        for input_dir in input_dirs:
            if not os.path.exists(input_dir):
                print(f"⚠️  Warning: Input directory not found: {input_dir}")
                continue
            
            label = os.path.basename(input_dir.rstrip("/\\"))
            output_dir = os.path.join(output_root, label)
            os.makedirs(output_dir, exist_ok=True)
            
            # Find all video files
            video_files = []
            for ext in video_exts:
                video_files.extend(Path(input_dir).glob(f"*{ext}"))
                video_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
            
            video_files = list(set(video_files))
            print(f"📁 Processing '{label}': {len(video_files)} videos found")
            
            # Prepare arguments for parallel processing
            process_args = []
            video_counter = 1
            for video_path in video_files:
                process_args.append((video_path, output_dir, config, label, video_counter))
                video_counter += 1
            
            # Process videos in parallel with progress bar
            if num_workers > 1:
                with Pool(processes=num_workers) as pool:
                    results = list(tqdm(
                        pool.imap(process_single_video, process_args),
                        total=len(process_args),
                        desc=f"  {split}/{label}"
                    ))
            else:
                # Sequential processing if num_workers = 1
                results = list(tqdm(
                    map(process_single_video, process_args),
                    total=len(process_args),
                    desc=f"  {split}/{label}"
                ))
            
            # Count results
            for status, error_msg, video_path in results:
                if status == "skip":
                    total_skipped += 1
                elif status == "success":
                    total_processed += 1
                else:
                    total_failed += 1
                    if error_msg:
                        print(f"\n⚠️  {error_msg}")
        
        print(f"\n✅ {split.upper()} - Processed: {total_processed}, Skipped: {total_skipped}, Failed: {total_failed}")
        print(f"📂 Output: {output_root}")
        
        total_processed_all += total_processed
        total_failed_all += total_failed
        total_skipped_all += total_skipped
    
    # Final summary
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
