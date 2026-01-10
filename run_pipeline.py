import subprocess
import sys
import time
from datetime import datetime

# -------------------------------
# Configuration
# -------------------------------
SCRIPTS = [
    {
        "name": "Video Preprocessing",
        "script": "preprocess_videos.py",
        "description": "Extracting frames from videos"
    },
    {
        "name": "Embedding Extraction",
        "script": "extract_embeddings.py",
        "description": "Generating DINOv3 embeddings from frames"
    },
    {
        "name": "Model Training",
        "script": "train_temporal_cnn.py",
        "description": "Training Temporal CNN classifier"
    }
]

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_step(step_num, total_steps, name, description):
    """Print step information"""
    print(f"\n{'#' * 70}")
    print(f"# STEP {step_num}/{total_steps}: {name}")
    print(f"# {description}")
    print(f"{'#' * 70}\n")

def run_script(script_path):
    """
    Run a Python script and return success status
    
    Args:
        script_path: Path to the Python script
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Run the script using the same Python interpreter
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True,
            capture_output=False  # Show output in real-time
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Script '{script_path}' failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Error: Script '{script_path}' not found")
        return False
    except Exception as e:
        print(f"\n❌ Error: Unexpected error running '{script_path}': {e}")
        return False

def main():
    """Main pipeline execution"""
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print_header("🚀 DEEPFAKE DETECTION PIPELINE")
    print(f"Started at: {start_datetime}")
    print(f"Total steps: {len(SCRIPTS)}")
    
    successful_steps = 0
    
    for idx, script_info in enumerate(SCRIPTS, 1):
        print_step(idx, len(SCRIPTS), script_info["name"], script_info["description"])
        
        step_start = time.time()
        success = run_script(script_info["script"])
        step_duration = time.time() - step_start
        
        if success:
            print(f"\n✅ Step {idx} completed successfully in {step_duration:.2f} seconds")
            successful_steps += 1
        else:
            print(f"\n❌ Step {idx} failed after {step_duration:.2f} seconds")
            print(f"Pipeline aborted at step {idx}/{len(SCRIPTS)}")
            break
    
    # Final summary
    total_duration = time.time() - start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print_header("📊 PIPELINE SUMMARY")
    print(f"Started:  {start_datetime}")
    print(f"Finished: {end_datetime}")
    print(f"Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"\nSteps completed: {successful_steps}/{len(SCRIPTS)}")
    
    if successful_steps == len(SCRIPTS):
        print("\n🎉 All steps completed successfully!")
        print("=" * 70)
        return 0
    else:
        print("\n⚠️  Pipeline incomplete - some steps failed")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
