#!/usr/bin/env python3
import cv2
import os
import sys
import argparse
import logging
import time
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GunnyBagCounter")

# Add the current directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import YOLO model directly from ultralytics
try:
    from ultralytics import YOLO
    DIRECT_YOLO = True
    logger.info("Using direct YOLO import from ultralytics")
except ImportError:
    DIRECT_YOLO = False
    # Import the YOLOModel from the app
    from app.ai_engine.models.yolo_model import YOLOModel
    logger.info("Using YOLOModel from app.ai_engine.models")

class GunnyBagVideoCounter:
    """
    A standalone application that opens a video file, counts gunny bags,
    and displays the count as an overlay on the video in real-time.
    """
    def __init__(self, model_path="app/yolov8n.pt", output_dir="data/gunny_bags", is_custom_model=False):
        """
        Initialize the counter
        
        Args:
            model_path: Path to the YOLO model file
            output_dir: Directory to save output files
            is_custom_model: Whether the model is a custom-trained model for gunny bags
        """
        # Initialize YOLO model
        logger.info(f"Loading YOLO model from {model_path}...")
        
        self.is_custom_model = is_custom_model
        
        # Load the model using the appropriate method
        if DIRECT_YOLO:
            # Load directly with ultralytics YOLO
            self.model = YOLO(model_path)
            logger.info(f"Model loaded directly with ultralytics YOLO")
            
            # Print model info for debugging
            try:
                logger.info(f"Model info: {len(self.model.names)} classes: {self.model.names}")
            except Exception as e:
                logger.warning(f"Could not print model info: {str(e)}")
        else:
            # Use the app's YOLOModel
            self.model = YOLOModel(model_path)
            logger.info(f"Model loaded with app's YOLOModel")
        
        # Configure output directory
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Set class IDs for detection - use a LOWER confidence threshold
        # 0.1 instead of 0.3 to catch more potential detections
        self.confidence_threshold = 0.1
        
        if is_custom_model:
            # For custom trained model, gunny bag is class 0
            self.target_classes = [0]
            logger.info("Using custom trained gunny bag model")
        else:
            # For default model, use similar object classes
            self.target_classes = [24, 26, 28, 39, 64]  # backpack, suitcase, handbag, bottle, etc.
            logger.info("Using default YOLO model with similar object classes")
        
        # Overlay properties
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.font_thickness = 2
        self.text_color = (255, 255, 255)
        self.bg_color = (0, 120, 0)
        
        # Tracking variables
        self.total_frames = 0
        self.processed_frames = 0
        self.detection_history = []
        self.max_history = 10  # Keep track of the last 10 frames for smoothing
        self.start_time = None
        
        logger.info("GunnyBagVideoCounter initialized")
        
    def detect_gunny_bags(self, frame):
        """
        Detect gunny bags in a frame
        
        Args:
            frame: Video frame to process
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Run YOLOv8 detection
            if DIRECT_YOLO:
                results = self.model(frame, verbose=False)
            else:
                if self.model is None or self.model.model is None:
                    return []
                results = self.model.model(frame, verbose=False)
            
            # Process results
            detections = []
            
            for result in results:
                boxes = result.boxes
                # Debug: print raw detection results
                logger.debug(f"Raw detections: {len(boxes)} boxes detected")
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get confidence
                    confidence = float(box.conf[0])
                    
                    # Get class
                    class_id = int(box.cls[0])
                    
                    # Debug: log all detections regardless of confidence
                    if confidence > 0.05:  # Super low threshold just for debugging
                        logger.debug(f"Detection: class={class_id}, confidence={confidence:.2f}")
                    
                    # Set proper class name
                    if self.is_custom_model:
                        class_name = "Gunny Bag"
                    else:
                        class_names = {
                            24: "Backpack (Gunny)", 
                            26: "Handbag (Gunny)", 
                            28: "Suitcase (Gunny)", 
                            39: "Bottle (Gunny)",
                            64: "Plant (Gunny)"
                        }
                        class_name = class_names.get(class_id, f"Class {class_id}")
                    
                    # In custom model mode, accept class 0 with any confidence above threshold
                    # In default model mode, filter for specific classes
                    accept_detection = False
                    
                    if self.is_custom_model:
                        # For custom model, accept class 0 (gunny bag) with confidence above threshold
                        if confidence > self.confidence_threshold:
                            accept_detection = True
                    else:
                        # For default model, filter for target classes
                        if class_id in self.target_classes and confidence > self.confidence_threshold:
                            accept_detection = True
                    
                    # If accepted, add to detections
                    if accept_detection:
                        detection = {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": class_name
                        }
                        detections.append(detection)
            
            if detections:
                logger.info(f"Detected {len(detections)} gunny bags in frame")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting gunny bags: {str(e)}")
            return []
    
    def process_video(self, video_path, display=True, save_output=True):
        """
        Process a video file, displaying and optionally saving the results
        
        Args:
            video_path: Path to the video file
            display: Whether to display the video
            save_output: Whether to save the processed video
            
        Returns:
            Dictionary with processing results
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {"success": False, "error": "Video file not found"}
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return {"success": False, "error": "Failed to open video file"}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {frame_width}x{frame_height}, {fps} fps, {self.total_frames} frames")
            
            # Create video writer if saving output
            output_path = None
            writer = None
            
            if save_output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"gunny_bags_counted_{timestamp}.mp4"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Define the codec and create VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                logger.info(f"Saving output to {output_path}")
            
            # Start processing timer
            self.start_time = time.time()
            self.processed_frames = 0
            frame_count = 0
            
            # Track max count
            max_count = 0
            
            # Process frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every frame
                frame_count += 1
                self.processed_frames += 1
                
                # Detect gunny bags in the frame
                detections = self.detect_gunny_bags(frame)
                count = len(detections)
                
                # Update history for smoothing
                self.detection_history.append(count)
                if len(self.detection_history) > self.max_history:
                    self.detection_history.pop(0)
                
                # Calculate smoothed count (moving average)
                smoothed_count = int(round(sum(self.detection_history) / len(self.detection_history)))
                
                # Update max count
                if smoothed_count > max_count:
                    max_count = smoothed_count
                
                # Draw overlay and detections
                result_frame = self._draw_overlay(frame, detections, smoothed_count, max_count, frame_count)
                
                # Save frame to video if requested
                if writer is not None:
                    writer.write(result_frame)
                
                # Display the frame if requested
                if display:
                    cv2.imshow('Gunny Bag Counter', result_frame)
                    
                    # Control playback speed
                    key = cv2.waitKey(1)  # Wait 1ms between frames
                    if key == 27:  # ESC key to exit
                        break
                    elif key == ord(' '):  # Spacebar to pause/resume
                        cv2.waitKey(0)
            
            # Calculate processing statistics
            elapsed_time = time.time() - self.start_time
            processing_fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0
            
            # Clean up
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            
            logger.info(f"Video processing completed: {self.processed_frames}/{self.total_frames} frames")
            logger.info(f"Processing speed: {processing_fps:.2f} fps")
            logger.info(f"Maximum gunny bag count: {max_count}")
            
            return {
                "success": True,
                "max_count": max_count,
                "frames_processed": self.processed_frames,
                "total_frames": self.total_frames,
                "processing_time": elapsed_time,
                "processing_fps": processing_fps,
                "output_path": output_path
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _draw_overlay(self, frame, detections, current_count, max_count, frame_number):
        """
        Draw overlay with detection information on the frame
        
        Args:
            frame: Video frame
            detections: List of detection dictionaries
            current_count: Current bag count
            max_count: Maximum bag count seen so far
            frame_number: Current frame number
            
        Returns:
            Frame with overlay
        """
        result_frame = frame.copy()
        
        # Draw bounding boxes around detections
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            class_name = detection.get("class_name", "Gunny Bag")
            
            # Draw rectangle
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            text_size = cv2.getTextSize(label, self.font, 0.5, 1)[0]
            cv2.rectangle(result_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), self.font, 0.5, (0, 0, 0), 1)
        
        # Calculate processing statistics
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            fps = self.processed_frames / elapsed_time if elapsed_time > 0 else 0
            progress = self.processed_frames / self.total_frames if self.total_frames > 0 else 0
            
            # Format time remaining
            if progress > 0:
                time_remaining = elapsed_time * (1 / progress - 1)
                mins = int(time_remaining // 60)
                secs = int(time_remaining % 60)
                time_str = f"{mins}m {secs}s remaining"
            else:
                time_str = "Calculating..."
        else:
            fps = 0
            progress = 0
            time_str = "Starting..."
        
        # Draw main overlay box at the top
        overlay_height = 100
        margin = 10
        
        # Create semi-transparent overlay
        overlay = result_frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)
        
        # Draw title
        cv2.putText(result_frame, "Gunny Bag Counter", 
                  (margin, margin + 25), self.font, 1.0, (0, 255, 255), 2)
        
        # Draw count information
        count_text = f"Current Count: {current_count}   Max Count: {max_count}"
        cv2.putText(result_frame, count_text, 
                  (margin, margin + 60), self.font, 0.8, (255, 255, 255), 2)
        
        # Draw confidence threshold info
        threshold_text = f"Confidence Threshold: {self.confidence_threshold:.2f}"
        cv2.putText(result_frame, threshold_text,
                  (margin, margin + 85), self.font, 0.6, (180, 180, 180), 1)
        
        # Draw processing information at the bottom
        info_y = frame.shape[0] - margin - 10
        
        # Create semi-transparent overlay for bottom info
        overlay = result_frame.copy()
        cv2.rectangle(overlay, (0, info_y - 30), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)
        
        # Draw frame info
        model_type = "Custom Model" if self.is_custom_model else "Default Model"
        frame_info = f"Frame: {frame_number}/{self.total_frames} ({int(progress*100)}%)  Processing: {fps:.1f} fps  {time_str}  Model: {model_type}"
        cv2.putText(result_frame, frame_info, 
                  (margin, info_y), self.font, 0.6, (180, 180, 180), 1)
        
        # Draw progress bar
        bar_width = frame.shape[1] - 2 * margin
        bar_height = 5
        bar_y = info_y - 15
        
        # Background
        cv2.rectangle(result_frame, (margin, bar_y), (margin + bar_width, bar_y + bar_height), (70, 70, 70), -1)
        
        # Foreground (progress)
        progress_width = int(bar_width * progress)
        cv2.rectangle(result_frame, (margin, bar_y), (margin + progress_width, bar_y + bar_height), (0, 255, 0), -1)
        
        return result_frame


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Gunny Bag Counter with Video Overlay")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--no-display", action="store_true", help="Don't display the video (headless mode)")
    parser.add_argument("--no-save", action="store_true", help="Don't save the processed video")
    parser.add_argument("--output-dir", default="data/gunny_bags", help="Directory to save output files")
    parser.add_argument("--model", default="app/yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--threshold", type=float, default=0.1, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger("GunnyBagCounter").setLevel(logging.DEBUG)
    
    # Check if using custom trained model
    is_custom_model = "gunny_bag_detector" in args.model or "trained_models" in args.model or "best.pt" in args.model or "train" in args.model
    
    # Initialize and run the counter
    counter = GunnyBagVideoCounter(
        model_path=args.model, 
        output_dir=args.output_dir,
        is_custom_model=is_custom_model
    )
    
    # Set confidence threshold from command line
    counter.confidence_threshold = args.threshold
    
    result = counter.process_video(
        args.video_path,
        display=not args.no_display,
        save_output=not args.no_save
    )
    
    if result["success"]:
        logger.info("✅ Processing completed successfully")
        logger.info(f"Maximum gunny bag count: {result['max_count']}")
        if not args.no_save and result["output_path"]:
            logger.info(f"Output saved to: {result['output_path']}")
    else:
        logger.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())