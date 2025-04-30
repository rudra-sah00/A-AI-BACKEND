import cv2
import logging
import os
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import YOLO model
from app.ai_engine.models.yolo_model import YOLOModel

logger = logging.getLogger(__name__)

class GunnyBagCounter:
    """
    A simple processor that counts gunny bags in videos using YOLOv8
    """
    
    def __init__(self, output_dir: str = None):
        """Initialize the gunny bag counter"""
        # Initialize YOLO model
        self.model = YOLOModel("yolov8n.pt")  # Using the default model
        
        # Configure output directory for results
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Set class ids for detection (will be updated when custom model is used)
        # For now using default classes that might look like gunny bags (backpack, suitcase, etc.)
        self.target_classes = [24, 26, 28, 39, 64]  # Backpack, suitcase, handbag, bottle, etc.
        
        logger.info("GunnyBagCounter initialized")
    
    def count_in_video(self, video_path: str) -> Dict[str, Any]:
        """
        Count gunny bags in a video file using YOLOv8
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dict with results including count
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {
                "success": False,
                "error": "Video file not found"
            }
            
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {
                    "success": False,
                    "error": "Failed to open video file"
                }
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process video at intervals for efficiency
            # Sample 1 frame per second
            sample_interval = max(1, int(fps))
            
            # Track detections
            frame_results = []
            processed_frames = 0
            best_frame = None
            best_frame_count = 0
            best_frame_detections = []
            
            # Process frames
            while True:
                # Read a frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every nth frame
                if processed_frames % sample_interval == 0:
                    # Detect objects using YOLO
                    results = self.detect_gunny_bags(frame)
                    count = len(results)
                    
                    # Store results
                    frame_results.append(count)
                    
                    # Keep track of the frame with the highest count
                    if count > best_frame_count:
                        best_frame_count = count
                        best_frame = frame.copy()
                        best_frame_detections = results
                    
                    logger.debug(f"Frame {processed_frames}: {count} gunny bags detected")
                
                processed_frames += 1
                
                # Process at most 300 frames (about 5 minutes of video at 1 fps)
                if processed_frames >= 300 * sample_interval:
                    break
            
            # Clean up
            cap.release()
            
            # Calculate results
            if frame_results:
                avg_count = sum(frame_results) / len(frame_results)
                max_count = max(frame_results)
                
                # Save the best frame with annotations if output directory exists
                result_image_path = None
                if self.output_dir and best_frame is not None:
                    # Draw bounding boxes on the best frame
                    for detection in best_frame_detections:
                        bbox = detection["bbox"]
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add label
                        label = f"Gunny Bag: {detection['confidence']:.2f}"
                        cv2.putText(best_frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add summary text
                    cv2.putText(best_frame, f"Total Count: {best_frame_count}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Save the image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result_image_path = os.path.join(self.output_dir, f"gunny_bags_{timestamp}.jpg")
                    cv2.imwrite(result_image_path, best_frame)
                
                return {
                    "success": True,
                    "count": max_count,
                    "avg_count": avg_count,
                    "frames_processed": len(frame_results),
                    "result_image": result_image_path,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": True,
                    "count": 0,
                    "message": "No gunny bags detected in video"
                }
        
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing video: {str(e)}"
            }
    
    def detect_gunny_bags(self, frame) -> List[Dict]:
        """
        Detect gunny bags in a frame using YOLO
        
        Args:
            frame: Image frame to process
            
        Returns:
            List of detections
        """
        if self.model is None or self.model.model is None:
            return []
        
        try:
            # Run detection without filtering for specific class
            # This will detect all objects and we'll filter
            results = self.model.model(frame, verbose=False)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get confidence
                    confidence = float(box.conf[0])
                    
                    # Get class
                    class_id = int(box.cls[0])
                    
                    # Filter for target classes
                    # In a production system, you would train a custom model
                    # specifically for gunny bags
                    if class_id in self.target_classes and confidence > 0.4:
                        detection = {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": confidence,
                            "class_id": class_id
                        }
                        detections.append(detection)
            
            return detections
        
        except Exception as e:
            logger.error(f"Error detecting gunny bags: {str(e)}")
            return []