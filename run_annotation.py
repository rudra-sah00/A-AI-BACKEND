#!/usr/bin/env python3
import os
import subprocess
import sys

def check_labelimg_installed():
    """Check if LabelImg is installed"""
    try:
        import labelImg
        return True
    except ImportError:
        return False

def install_labelimg():
    """Install LabelImg package"""
    print("Installing LabelImg...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "labelImg"])
        print("LabelImg installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Error installing LabelImg. Try installing manually with:")
        print("pip install labelImg")
        return False

def update_classes_file():
    """Ensure the classes file contains both moving and static gunny bag classes"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    classes_file = os.path.join(current_dir, "data", "gunny_bag_dataset", "classes.txt")
    classes = ["gunny_bag_counting", "gunny_bag_ignore"]
    
    # Create or update the classes file
    with open(classes_file, 'w') as f:
        f.write('\n'.join(classes))
    
    print(f"Updated classes file at {classes_file} with classes: {', '.join(classes)}")

def run_annotation():
    # Check if LabelImg is installed
    if not check_labelimg_installed():
        success = install_labelimg()
        if not success:
            return 1
    
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "data", "gunny_bag_dataset", "images")
    classes_file = os.path.join(current_dir, "data", "gunny_bag_dataset", "classes.txt")
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return 1
    
    # Update classes file to include both types of gunny bags
    update_classes_file()
    
    # Create labels directory if it doesn't exist
    labels_dir = os.path.join(current_dir, "data", "gunny_bag_dataset", "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    # Print annotation instructions
    print("\n=== Gunny Bag Counter - Annotation Tool ===")
    print(f"\nStarting LabelImg with:")
    print(f"- Images directory: {images_dir}")
    print(f"- Classes file: {classes_file}")
    print(f"- Labels will be saved to: {labels_dir}")
    print("\nAnnotation Instructions:")
    print("1. Use 'W' to create a bounding box around each gunny bag")
    print("2. Label each bag based on its purpose:")
    print("   - Use 'gunny_bag_counting' for bags you want the system to count (on the machine)")
    print("   - Use 'gunny_bag_ignore' for bags you DON'T want counted (kept aside)")
    print("3. Press 'Ctrl+S' to save each annotation")
    print("4. Use 'D' for next image, 'A' for previous image")
    print("5. Make sure to set the save format to YOLO in View > Auto Save Mode > YOLO")
    print("\nWhy annotate both types?")
    print("- This teaches the model to recognize which bags to count and which to ignore")
    print("- The counting code will only count 'gunny_bag_counting' class detections")
    print("- This approach prevents false counting of static bags")
    
    # Command to run LabelImg
    cmd = ["labelImg", images_dir, classes_file, labels_dir]
    
    try:
        print("\nLaunching LabelImg...")
        subprocess.run(cmd)
        print("\nAnnotation complete!")
        print(f"Label files have been saved to: {labels_dir}")
        print("\nNext step: Run the training script to train your YOLO model")
    except Exception as e:
        print(f"Error starting LabelImg: {e}")
        # Create a shell script for easy running
        script_name = "run_labelimg.sh"
        with open(script_name, "w") as f:
            f.write(f"#!/bin/bash\nlabelImg {images_dir} {classes_file} {labels_dir}\n")
        os.chmod(script_name, 0o755)
        print(f"Try running the convenience script: ./{script_name}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run_annotation())