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
    def validate_rtsp_stream(rtsp_url: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Validates an RTSP stream by:
        1. Attempting to connect to the stream
        2. Capturing a frame
        3. Verifying frame validity
        
        Args:
            rtsp_url: The RTSP URL to validate
            timeout: Maximum time (in seconds) to wait for connection
        
        Returns:
            Dictionary with validation results
        """
        start_time = time.time()
        result = {
            "is_valid": False,
            "message": "Failed to validate stream",
            "frame_width": 0,
            "frame_height": 0,
            "response_time": 0,
            "timestamp": time.time()
        }
        
        try:
            logger.info(f"Validating RTSP stream: {rtsp_url}")
            
            # Use the FFmpeg warning suppressor
            with ffmpeg_suppressor:
                # Additional OpenCV parameters to improve connection reliability
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Buffer size
                
                # Set connection timeout
                elapsed_time = 0
                connection_timeout = timeout
                
                # Try to connect until timeout
                while elapsed_time < connection_timeout:
                    if cap.isOpened():
                        break
                    time.sleep(0.5)
                    elapsed_time = time.time() - start_time
                    
                if not cap.isOpened():
                    result["message"] = f"Could not connect to stream within {timeout} seconds"
                    return result
                    
                # Try to read a frame
                ret, frame = cap.read()
            
            # Calculate response time
            response_time = time.time() - start_time
            result["response_time"] = round(response_time, 2)
            
            if not ret or frame is None:
                result["message"] = "Connected to stream but could not read frame"
                return result
                
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            result["frame_width"] = frame_width
            result["frame_height"] = frame_height
            
            # Check if frame has valid dimensions
            if frame_width <= 0 or frame_height <= 0:
                result["message"] = "Received invalid frame dimensions"
                return result
                
            # Success
            result["is_valid"] = True
            result["message"] = "Stream validated successfully"
            
            logger.info(f"RTSP stream validation successful: {rtsp_url}")
            
        except Exception as e:
            logger.error(f"Error validating RTSP stream: {str(e)}")
            result["message"] = f"Error: {str(e)}"
        finally:
            # Release the capture object
            if 'cap' in locals() and cap is not None:
                cap.release()
                
        return result
    
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