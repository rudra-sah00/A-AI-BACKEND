#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
import shutil
import random
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("YOLOTrainer")

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Ultralytics package not found. Install with: pip install ultralytics")
    exit(1)

def create_dataset_yaml():
    """Create the dataset.yaml file needed for YOLOv8 training"""
    current_dir = Path(__file__).parent.absolute()
    dataset_dir = current_dir / "data" / "gunny_bag_dataset"
    
    # Read class names
    classes_file = dataset_dir / "classes.txt"
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Create train/val splits
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    random.shuffle(image_files)
    
    # Create train/val directories
    train_dir = dataset_dir / "train" / "images"
    val_dir = dataset_dir / "val" / "images"
    train_labels_dir = dataset_dir / "train" / "labels"
    val_labels_dir = dataset_dir / "val" / "labels"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Split data: 80% train, 20% validation
    split_idx = int(len(image_files) * 0.8)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    # Copy images and corresponding labels to train/val directories
    for img_file in train_images:
        # Copy image
        shutil.copy(img_file, train_dir / img_file.name)
        # Look for corresponding label file
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy(label_file, train_labels_dir / label_file.name)
    
    for img_file in val_images:
        # Copy image
        shutil.copy(img_file, val_dir / img_file.name)
        # Look for corresponding label file
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy(label_file, val_labels_dir / label_file.name)
    
    # Create dataset.yaml file
    dataset_yaml = {
        'path': str(dataset_dir),
        'train': str(train_dir.parent),
        'val': str(val_dir.parent),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"Created dataset YAML at {yaml_path}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Classes: {class_names}")
    
    return yaml_path

def train_yolo_model(data_yaml, model_name="yolov8n.pt", epochs=50, batch_size=16, imgsz=640):
    """
    Train a YOLO model on custom data
    
    Args:
        data_yaml: Path to the data.yaml file
        model_name: Base model to use for training
        epochs: Number of training epochs
        batch_size: Batch size for training
        imgsz: Image size for training
    
    Returns:
        Path to the trained model
    """
    logger.info(f"Starting YOLO training with {model_name} for {epochs} epochs")
    
    try:
        # Load the base model
        model = YOLO(model_name)
         
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            patience=15,  # Increased patience for better convergence
            save=True,
            device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            # Enhanced data augmentation settings
            mosaic=1.0,          # Increase mosaic augmentation probability
            mixup=0.2,           # Enable mixup augmentation
            degrees=15.0,        # Rotation augmentation
            translate=0.2,       # Translation augmentation
            scale=0.8,           # More aggressive scaling
            shear=5.0,           # Shear augmentation
            perspective=0.0005,  # Perspective transform
            flipud=0.2,          # Flip up-down
            fliplr=0.5,          # Flip left-right
            hsv_h=0.015,         # HSV hue augmentation
            hsv_s=0.7,           # HSV saturation augmentation
            hsv_v=0.4,           # HSV value augmentation
            copy_paste=0.2       # Copy-paste augmentation
        )
        
        # Get the path to the best model - the attribute access has changed in newer versions
        # Instead of trying to access results.best, we can use a more reliable path
        run_dir = Path(results.save_dir)
        best_model_path = run_dir / "weights" / "best.pt"
        
        if best_model_path.exists():
            logger.info(f"Training completed. Best model saved to {best_model_path}")
        else:
            # Fallback to last.pt if best.pt doesn't exist for some reason
            best_model_path = run_dir / "weights" / "last.pt"
            logger.info(f"Training completed. Using last model saved to {best_model_path}")
        
        # Copy the best model to a more accessible location
        output_dir = Path("trained_models")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"gunny_bag_detector.pt"
        shutil.copy2(best_model_path, output_path)
        
        logger.info(f"Best model copied to {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return None

def auto_annotate_data(data_yaml, model_name="yolov8n.pt"):
    """
    Attempt to auto-annotate the dataset using a pre-trained model
    
    Args:
        data_yaml: Path to the data.yaml file
        model_name: Model to use for annotation
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Attempting to auto-annotate the dataset with a pre-trained model")
    
    try:
        # Load the data.yaml file
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Get the image directories
        dataset_path = Path(data_config.get('path', '.'))
        train_images_dir = dataset_path / data_config.get('train', 'images/train')
        val_images_dir = dataset_path / data_config.get('val', 'images/val')
        
        # Get the label directories
        train_labels_dir = dataset_path / 'labels/train'
        val_labels_dir = dataset_path / 'labels/val'
        
        # Load model for auto-annotation
        model = YOLO(model_name)
        
        # Define the class mappings
        # Mapping from COCO classes to our gunny bag class
        # 24: backpack, 26: handbag, 28: suitcase, 39: bottle, 64: potted plant
        class_mappings = {24: 0, 26: 0, 28: 0, 39: 0, 64: 0}
        
        # Process training images
        logger.info("Auto-annotating training images...")
        for img_path in train_images_dir.glob("*.jpg"):
            # Run prediction on image
            results = model.predict(img_path, conf=0.25)
            
            # Create annotation file
            label_path = train_labels_dir / f"{img_path.stem}.txt"
            
            with open(label_path, 'w') as f:
                # Process each detection
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get class ID and check if it should be mapped
                        cls_id = int(box.cls[0].item())
                        if cls_id in class_mappings:
                            # Convert to x, y, width, height format (normalized)
                            xyxy = box.xyxy[0].tolist()  # x1, y1, x2, y2
                            img_width, img_height = result.orig_shape[1], result.orig_shape[0]
                            
                            # Convert to YOLOv8 format (x_center, y_center, width, height) - normalized
                            x_center = ((xyxy[0] + xyxy[2]) / 2) / img_width
                            y_center = ((xyxy[1] + xyxy[3]) / 2) / img_height
                            width = (xyxy[2] - xyxy[0]) / img_width
                            height = (xyxy[3] - xyxy[1]) / img_height
                            
                            # Write to file
                            f.write(f"{class_mappings[cls_id]} {x_center} {y_center} {width} {height}\n")
        
        # Process validation images
        logger.info("Auto-annotating validation images...")
        for img_path in val_images_dir.glob("*.jpg"):
            # Run prediction on image
            results = model.predict(img_path, conf=0.25)
            
            # Create annotation file
            label_path = val_labels_dir / f"{img_path.stem}.txt"
            
            with open(label_path, 'w') as f:
                # Process each detection
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get class ID and check if it should be mapped
                        cls_id = int(box.cls[0].item())
                        if cls_id in class_mappings:
                            # Convert to x, y, width, height format (normalized)
                            xyxy = box.xyxy[0].tolist()  # x1, y1, x2, y2
                            img_width, img_height = result.orig_shape[1], result.orig_shape[0]
                            
                            # Convert to YOLOv8 format (x_center, y_center, width, height) - normalized
                            x_center = ((xyxy[0] + xyxy[2]) / 2) / img_width
                            y_center = ((xyxy[1] + xyxy[3]) / 2) / img_height
                            width = (xyxy[2] - xyxy[0]) / img_width
                            height = (xyxy[3] - xyxy[1]) / img_height
                            
                            # Write to file
                            f.write(f"{class_mappings[cls_id]} {x_center} {y_center} {width} {height}\n")
        
        logger.info("Auto-annotation completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error during auto-annotation: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model for gunny bag detection")
    parser.add_argument("--data", help="Path to the data.yaml file")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model to use for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--auto-annotate", action="store_true", help="Automatically annotate using pre-trained model")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare dataset.yaml without training")
    args = parser.parse_args()
    
    # If prepare-only is set, just create the dataset YAML and exit
    if args.prepare_only:
        yaml_path = create_dataset_yaml()
        logger.info(f"Dataset YAML created at: {yaml_path}")
        return 0
    
    # Make sure the data.yaml file exists
    if not args.data:
        # If no data file specified, try to create one
        yaml_path = create_dataset_yaml()
        args.data = str(yaml_path)
    elif not os.path.exists(args.data):
        logger.error(f"Data YAML file not found: {args.data}")
        return 1
    
    # Auto-annotate data if requested
    if args.auto_annotate:
        if not auto_annotate_data(args.data, args.model):
            logger.error("Auto-annotation failed. Please annotate the dataset manually.")
            return 1
    
    # Train the model
    trained_model_path = train_yolo_model(args.data, args.model, args.epochs, args.batch, args.imgsz)
    
    if trained_model_path:
        logger.info("\n" + "-"*50)
        logger.info("Training completed successfully!")
        logger.info(f"Trained model saved to: {trained_model_path}")
        logger.info("\nTo use the model for inference:")
        logger.info(f"python gunny_bag_counter_app.py gunny-bag-test.mp4 --model {trained_model_path}")
        logger.info("-"*50)
        return 0
    else:
        logger.error("Training failed.")
        return 1

if __name__ == "__main__":
    exit(main())