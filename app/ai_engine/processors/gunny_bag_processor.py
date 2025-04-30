import cv2
import logging
import os
import time
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

from app.ai_engine.processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

class GunnyBagProcessor(BaseProcessor):
    """
    Processor for counting gunny bags in video frames
    This is a test implementation for API testing purposes
    """
    def __init__(self, camera_data: Dict, users_data: Dict, output_dir: str):
        # Initialize with empty rule_data since this is just a test processor
        super().__init__({}, camera_data, users_data, output_dir)
        
        # Configuration for gunny bag detection
        self.min_area = 1000  # Minimum contour area to be considered a gunny bag
        self.max_area = 100000  # Maximum contour area
        self.detection_threshold = 0.6  # Confidence threshold
        
        # Store the last detection results
        self.last_result = {
            "timestamp": None,
            "count": 0,
            "image_path": None,
            "confidence": 0.0
        }
        
        # Create output directory for saving detection results
        self.results_dir = os.path.join(output_dir, "gunny_bags")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Track active status
        self.is_active = False
        
        # Lock for thread-safe access to results
        self.result_lock = threading.Lock()
        
        logger.info(f"Initialized GunnyBagProcessor for camera {self.camera_data.get('name')}")
    
    def process(self):
        """Start processing the camera feed"""
        if self.running:
            return
        
        self.running = True
        
        # Connect to the camera
        connection_attempts = 0
        max_attempts = 3
        
        while connection_attempts < max_attempts:
            if self._connect_to_camera():
                break
            
            connection_attempts += 1
            if connection_attempts < max_attempts:
                logger.warning(f"Retrying camera connection (attempt {connection_attempts}/{max_attempts})")
                time.sleep(2)  # Wait before retrying
        
        if connection_attempts >= max_attempts:
            logger.error(f"Failed to connect to camera {self.camera_data.get('name')} after {max_attempts} attempts")
            self.running = False
            return
        
        # Update the last process time after successful connection
        self.last_process_time = time.time()
        
        # Start the processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.is_active = True
        logger.info(f"GunnyBagProcessor started for camera {self.camera_data.get('name')}")
    
    def stop(self):
        """Stop the processor"""
        self.running = False
        self.is_active = False
        
        # Release the camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        logger.info(f"GunnyBagProcessor stopped for camera {self.camera_data.get('name')}")
    
    def _processing_loop(self):
        """Main processing loop"""
        try:
            while self.running:
                try:
                    # Get a frame
                    frame = self._get_frame()
                    
                    if frame is not None:
                        self.last_frame = frame
                        self.last_process_time = time.time()
                        
                        # Process the frame
                        self._process_frame(frame)
                    
                    # Sleep to maintain target FPS
                    time.sleep(self.frame_interval)
                    
                except Exception as loop_error:
                    logger.error(f"Error in frame processing: {str(loop_error)}")
                    time.sleep(1)  # Sleep longer on error
        except Exception as e:
            logger.error(f"Fatal error in processing loop: {str(e)}")
            self.running = False
    
    def _process_frame(self, frame):
        """
        Process a frame to count gunny bags
        
        This is a simple implementation for testing that uses color thresholding
        and contour detection to identify potential gunny bags
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and shape
            gunny_bag_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area:
                    # Check aspect ratio (gunny bags are typically rectangular)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Typical gunny bags have aspect ratio between 0.5 and 2.0
                    if 0.5 < aspect_ratio < 2.0:
                        gunny_bag_contours.append(contour)
            
            # Count the detected gunny bags
            count = len(gunny_bag_contours)
            
            # Only update result if the count is different from the last one
            # or it's been more than 10 seconds since the last update
            with self.result_lock:
                current_time = time.time()
                last_update_time = self.last_result["timestamp"]
                time_since_last_update = float('inf') if last_update_time is None else current_time - last_update_time
                
                if self.last_result["count"] != count or time_since_last_update > 10:
                    # Draw contours on the frame
                    result_frame = frame.copy()
                    cv2.drawContours(result_frame, gunny_bag_contours, -1, (0, 255, 0), 2)
                    
                    # Add text with count
                    cv2.putText(
                        result_frame, f"Gunny Bags: {count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )
                    
                    # Add timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(
                        result_frame, timestamp,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    
                    # Save the result frame
                    filename = f"gunny_bags_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    image_path = os.path.join(self.results_dir, filename)
                    cv2.imwrite(image_path, result_frame)
                    
                    # Update the last result
                    self.last_result = {
                        "timestamp": current_time,
                        "count": count,
                        "image_path": image_path,
                        "confidence": 0.7,  # Mock confidence value
                        "frame_width": frame.shape[1],
                        "frame_height": frame.shape[0]
                    }
                    
                    logger.info(f"Updated gunny bag count: {count} for camera {self.camera_data.get('name')}")
            
            return count
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return None
    
    def get_latest_result(self):
        """Get the latest detection result"""
        with self.result_lock:
            if self.last_result["timestamp"] is not None:
                # Create a copy to avoid threading issues
                return dict(self.last_result)
            else:
                return {
                    "timestamp": None,
                    "count": 0,
                    "image_path": None,
                    "confidence": 0.0,
                    "message": "No detection results yet"
                }
    
    def process_video(self, video_path):
        """
        Process a video file to count gunny bags
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dict with results information
        """
        try:
            # Check if the file exists
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return {
                    "success": False,
                    "error": f"Video file not found: {video_path}"
                }
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return {
                    "success": False,
                    "error": f"Failed to open video file: {video_path}"
                }
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Process video in chunks for efficiency
            sample_interval = max(1, int(fps))  # Process 1 frame per second
            max_samples = 100  # Maximum number of frames to sample
            
            # Calculate frame indices to sample
            if frame_count > 0:
                sample_indices = [
                    int(i * frame_count / min(max_samples, frame_count / sample_interval))
                    for i in range(min(max_samples, int(frame_count / sample_interval)))
                ]
            else:
                sample_indices = []
            
            # Initialize results
            gunny_bag_counts = []
            result_frames = []
            
            for frame_idx in sample_indices:
                # Set the position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Read the frame
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Process the frame
                count = self._process_frame(frame)
                if count is not None:
                    gunny_bag_counts.append(count)
                    
                    # Store result frame (with contours and text)
                    with self.result_lock:
                        if self.last_result["image_path"] and os.path.exists(self.last_result["image_path"]):
                            result_frames.append(self.last_result["image_path"])
            
            # Calculate results
            if gunny_bag_counts:
                avg_count = sum(gunny_bag_counts) / len(gunny_bag_counts)
                max_count = max(gunny_bag_counts)
                
                # Save a summary result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"summary_gunny_bags_{timestamp}.jpg"
                summary_path = os.path.join(self.results_dir, filename)
                
                # Get the frame with the maximum count
                if result_frames:
                    best_frame = cv2.imread(result_frames[-1])
                    if best_frame is not None:
                        # Add summary text
                        cv2.putText(
                            best_frame, f"Average Count: {avg_count:.1f}, Max Count: {max_count}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )
                        cv2.imwrite(summary_path, best_frame)
                
                return {
                    "success": True,
                    "video_path": video_path,
                    "timestamp": datetime.now().isoformat(),
                    "avg_count": avg_count,
                    "max_count": max_count,
                    "counts": gunny_bag_counts,
                    "frames_processed": len(gunny_bag_counts),
                    "total_frames": frame_count,
                    "video_fps": fps,
                    "video_width": width,
                    "video_height": height,
                    "result_image": summary_path if os.path.exists(summary_path) else None
                }
            else:
                return {
                    "success": False,
                    "error": "No gunny bags detected in the video",
                    "video_path": video_path
                }
        
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing video: {str(e)}",
                "video_path": video_path
            }
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()