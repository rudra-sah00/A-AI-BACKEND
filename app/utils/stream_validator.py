import cv2
import numpy as np
import logging
import os
import time
import urllib.parse
import ctypes
from typing import Tuple, Dict, Any
import sys

from app.utils.ffmpeg_utils import ffmpeg_suppressor

logger = logging.getLogger(__name__)

# Create utils directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)


class StreamValidator:
    """
    RTSP stream validator and handler
    """
    
    @staticmethod
    def format_rtsp_url(rtsp_url: str, username: str = None, password: str = None) -> str:
        """Format RTSP URL with authentication if provided"""
        if not (username and password):
            return rtsp_url
            
        parsed_url = urllib.parse.urlparse(rtsp_url)
        netloc = f"{username}:{password}@{parsed_url.netloc}"
        parts = list(parsed_url)
        parts[1] = netloc
        return urllib.parse.urlunparse(tuple(parts))
    
    @staticmethod
    def validate_rtsp_stream(rtsp_url, timeout=5.0, retry_attempts=3):
        """
        Validate that an RTSP stream is accessible and can be read
        Implements retry logic and better error handling for unreliable networks
        
        Args:
            rtsp_url: RTSP URL to validate
            timeout: Maximum time (in seconds) to wait for stream connection
            retry_attempts: Number of retry attempts before giving up
            
        Returns:
            Dictionary with validation results
        """
        if not rtsp_url:
            return {"is_valid": False, "error_message": "No RTSP URL provided"}
            
        logger.info(f"Validating RTSP stream: {rtsp_url}")
        
        # Add transport protocol directly to URL as a query parameter
        rtsp_url_with_options = rtsp_url
        if "?" not in rtsp_url_with_options:
            rtsp_url_with_options += "?rtsp_transport=tcp"
        else:
            rtsp_url_with_options += "&rtsp_transport=tcp"
            
        # Try to open the stream multiple times
        for attempt in range(retry_attempts):
            try:
                # Open the stream with only two arguments (compatible with OpenCV 4.11.0)
                start_time = time.time()
                cap = cv2.VideoCapture(rtsp_url_with_options, cv2.CAP_FFMPEG)
                
                # Check if stream opened
                if not cap.isOpened():
                    if attempt < retry_attempts - 1:
                        logger.warning(f"Failed to open stream (attempt {attempt+1}/{retry_attempts}), retrying...")
                        time.sleep(1)  # Wait before retrying
                        continue
                    return {"is_valid": False, "error_message": "Failed to open stream"}
                
                # Try to read frames with timeout
                success = False
                frame_count = 0
                max_frames = 5  # Try to read multiple frames for better validation
                
                while time.time() - start_time < timeout and frame_count < max_frames:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        success = True
                        frame_count += 1
                    else:
                        # If we can't read a frame, try again
                        if time.time() - start_time < timeout:
                            time.sleep(0.1)  # Small delay before retry
                        else:
                            break
                
                # Clean up
                cap.release()
                
                if success:
                    elapsed_time = time.time() - start_time
                    logger.info(f"RTSP stream validated successfully ({frame_count} frames) in {elapsed_time:.2f}s")
                    
                    # Check if we got enough frames to consider the stream reliable
                    if frame_count < 3:
                        logger.warning(f"Stream validated but only {frame_count} frames read - might be unstable")
                    
                    return {
                        "is_valid": True, 
                        "frames_read": frame_count,
                        "validation_time": elapsed_time
                    }
                
                if attempt < retry_attempts - 1:
                    logger.warning(f"Failed to read frames (attempt {attempt+1}/{retry_attempts}), retrying...")
                    time.sleep(1)  # Wait before retrying
                    continue
                
                return {
                    "is_valid": False, 
                    "error_message": "Could not read frames from stream",
                    "frames_read": frame_count
                }
                
            except Exception as e:
                if attempt < retry_attempts - 1:
                    logger.warning(f"Stream validation error (attempt {attempt+1}/{retry_attempts}): {str(e)}, retrying...")
                    time.sleep(1)  # Wait before retrying
                else:
                    logger.error(f"Stream validation failed after {retry_attempts} attempts: {str(e)}")
                    return {"is_valid": False, "error_message": str(e)}
        
        return {"is_valid": False, "error_message": "Failed to validate stream after multiple attempts"}
    
    @staticmethod
    def add_camera(camera_name: str, rtsp_url: str, validate: bool = True) -> Dict[str, Any]:
        """
        Add a camera with optional stream validation
        
        Args:
            camera_name: Name of the camera
            rtsp_url: RTSP URL for the camera stream
            validate: Whether to validate the stream before adding
            
        Returns:
            Dictionary with operation results
        """
        result = {
            "success": False,
            "name": camera_name,
            "rtsp_url": rtsp_url,
            "message": "Failed to add camera",
            "validation_result": None
        }
        
        try:
            # Validate the stream if requested
            if validate:
                validation_result = StreamValidator.validate_rtsp_stream(rtsp_url)
                result["validation_result"] = validation_result
                
                if not validation_result["is_valid"]:
                    result["message"] = f"Camera stream validation failed: {validation_result['message']}"
                    return result
                    
            # If we reach here, either validation was successful or not requested
            logger.info(f"Adding camera {camera_name} with URL {rtsp_url}")
            
            result["success"] = True
            result["message"] = "Camera added successfully" + (" (validated)" if validate else "")
            
        except Exception as e:
            logger.error(f"Error adding camera: {str(e)}")
            result["message"] = f"Error adding camera: {str(e)}"
            
        return result