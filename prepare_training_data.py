#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path

def ensure_dependencies():
    """Make sure all required packages are installed"""
    try:
        import ultralytics
        print("✅ Ultralytics already installed")
    except ImportError:
        print("📦 Installing ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("✅ Ultralytics installed successfully")

def run_command(cmd, description):
    """Run a shell command with error handling"""
    print(f"\n🔄 {description}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Prepare and train a custom YOLO model for gunny bag detection")
    parser.add_argument("--video", default="gunny-bag-test.mp4", help="Input video file for frame extraction")
    parser.add_argument("--fps", type=float, default=1, help="Frames per second to extract")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip frame extraction if already done")
    parser.add_argument("--auto-annotate", action="store_true", help="Try auto-annotation (experimental)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()

    current_dir = Path(__file__).parent.absolute()
    dataset_dir = current_dir / "data" / "gunny_bag_dataset"
    images_dir = dataset_dir / "images"
    
    # Step 1: Ensure directories exist
    print("\n🔍 Checking directories...")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(dataset_dir / "labels", exist_ok=True)
    print("✅ Directories created")
    
    # Step 2: Check for and install dependencies
    ensure_dependencies()
    
    # Step 3: Extract frames from video if needed
    if not args.skip_extraction:
        if not os.path.exists(args.video):
            print(f"❌ Video file not found: {args.video}")
            return 1
            
        # Count existing frames
        existing_frames = len(list(images_dir.glob("*.jpg")))
        if existing_frames > 0:
            print(f"⚠️ Found {existing_frames} existing frames. Use --skip-extraction to skip extraction.")
            response = input("Do you want to overwrite existing frames? (y/n): ")
            if response.lower() != 'y':
                print("ℹ️ Skipping frame extraction")
            else:
                # Extract frames using ffmpeg
                extract_cmd = [
                    "ffmpeg", "-i", args.video, 
                    "-vf", f"fps={args.fps}", 
                    "-q:v", "1", 
                    str(images_dir / "frame_%04d.jpg"),
                    "-y"  # Overwrite existing files
                ]
                run_command(extract_cmd, "Frame extraction")
        else:
            # Extract frames using ffmpeg
            extract_cmd = [
                "ffmpeg", "-i", args.video, 
                "-vf", f"fps={args.fps}", 
                "-q:v", "1", 
                str(images_dir / "frame_%04d.jpg")
            ]
            run_command(extract_cmd, "Frame extraction")
    
    # Step 4: Update classes.txt file for our specific use case
    print("\n📝 Setting up annotation classes...")
    with open(dataset_dir / "classes.txt", "w") as f:
        f.write("gunny_bag_counting\ngunny_bag_ignore")
    print("✅ Created classes.txt with gunny_bag_counting and gunny_bag_ignore classes")
    
    # Step 5: Open annotation tool if labels don't exist or seem incomplete
    label_files = list(Path(dataset_dir / "labels").glob("*.txt"))
    frame_files = list(images_dir.glob("*.jpg"))
    
    if len(label_files) < len(frame_files) / 2:  # If less than half the frames have annotations
        print(f"\n⚠️ Only {len(label_files)} out of {len(frame_files)} frames have annotations")
        print("\n📋 Instructions for annotation:")
        print("1. Use 'W' to draw a bounding box around each gunny bag in the image")
        print("2. Label bags being counted (moving up on machine) as 'gunny_bag_counting'")
        print("3. Label bags that should be ignored as 'gunny_bag_ignore'")
        print("4. Press Ctrl+S to save after annotating each image")
        print("5. Navigate using 'D' (next) and 'A' (previous)")
        print("6. Make sure to set save format to YOLO (View > Auto Save Mode > YOLO)")
        
        # Run the annotation script
        run_cmd = [sys.executable, str(current_dir / "run_annotation.py")]
        run_command(run_cmd, "Starting annotation tool")
        
        print("\n⏸️ Continue when you've finished annotating")
        input("Press Enter when you've completed the annotation process...")
    
    # Step 6: Prepare dataset for training
    print("\n🔄 Preparing dataset for training...")
    yaml_path = current_dir / "data" / "gunny_bag_dataset" / "dataset.yaml"
    
    # Use built-in create_dataset_yaml if the file doesn't exist
    if not yaml_path.exists():
        # Create the dataset YAML file
        train_cmd = [
            sys.executable,
            str(current_dir / "train_yolo.py"),
            "--prepare-only"  # This doesn't exist yet, let's add it later
        ]
        
        # For now, let's create the YAML manually
        print("Creating dataset.yaml manually...")
        from train_yolo import create_dataset_yaml
        yaml_path = create_dataset_yaml()
    
    # Step 7: Auto-annotate if requested
    if args.auto_annotate:
        print("\n🤖 Trying auto-annotation (experimental)...")
        auto_annotate_cmd = [
            sys.executable,
            str(current_dir / "train_yolo.py"),
            "--data", str(yaml_path),
            "--auto-annotate"
        ]
        run_command(auto_annotate_cmd, "Auto-annotation")
    
    # Step 8: Train the model
    print("\n🚀 Ready to start training!")
    print("Training will use the following settings:")
    print(f"- Base model: yolov8n.pt")
    print(f"- Epochs: {args.epochs}")
    print(f"- Dataset: {yaml_path}")
    
    response = input("\nStart training now? (y/n): ")
    if response.lower() == 'y':
        train_cmd = [
            sys.executable,
            str(current_dir / "train_yolo.py"),
            "--data", str(yaml_path),
            "--epochs", str(args.epochs)
        ]
        run_command(train_cmd, "Training YOLO model")
        
        print("\n🎉 Custom training process completed!")
        print("\nNext steps:")
        print("1. Check the 'trained_models' directory for your trained model")
        print("2. Use the model with: python gunny_bag_counter_app.py gunny-bag-test.mp4 --model trained_models/gunny_bag_detector.pt")
    else:
        print("\nℹ️ Training skipped. Run the following command when you're ready to train:")
        print(f"python train_yolo.py --data {yaml_path} --epochs {args.epochs}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())