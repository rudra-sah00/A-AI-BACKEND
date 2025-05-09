import base64
import cv2
import json
import logging
import os
import requests
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import io
from PIL import Image

from app.core.config import settings
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)

class OllamaVisionProcessor(BaseProcessor):
    """
    Processor for handling contextual questions using vision models
    """
    def __init__(self, camera_data: Dict, users_data: Dict, output_dir: str):
        # Initialize with empty rule_data since this processor doesn't use rules
        super().__init__({}, camera_data, users_data, output_dir)
        
        # Gemini model configuration
        self.model = "gemini-2.0-flash"
        # Hard-coded API key for development stage
        self.api_key = "PLACE YOU API KEY HERE"
        logger.info(f"Using hard-coded API key for development: {self.api_key[:5]}...{self.api_key[-4:]}")
            
        self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        # Load prompts for the vision model
        self.prompts = self._load_prompts()
        
        # Query handling
        self.query_lock = threading.Lock()
        self.last_query_time = 0
        self.query_cooldown = 1  # Seconds between queries to prevent flooding
        self.query_timeout = 30  # Seconds to wait for a response before timing out
        
        # Create output directory for saving query results
        self.query_output_dir = os.path.join(output_dir, "queries")
        os.makedirs(self.query_output_dir, exist_ok=True)
        
        # Track active status
        self.is_active = False
        
        # Initialize last_process_time to current time to prevent immediate reconnection
        self.last_process_time = time.time()
        
        logger.info(f"Initialized Vision processor for camera {self.camera_data.get('name')} using {self.model}")
    
    def _load_prompts(self) -> Dict:
        """Load prompts from the ollamavision_prompts.json file"""
        try:
            prompts_file = os.path.join(settings.DATA_DIR, "ai_prompts", "ollamavision_prompts.json")
            if os.path.exists(prompts_file):
                with open(prompts_file, 'r') as f:
                    loaded_prompts = json.load(f)
                    # Ensure essential prompts have defaults if not in file
                    if "system_context" not in loaded_prompts:
                        loaded_prompts["system_context"] = (
                            "You are a precise visual analysis assistant. "
                            "Your task is to analyze the provided security camera image and answer the user's question. "
                            "Base your answer *only* on the visual information present in the image. "
                            "Be concise and direct. Do not speculate or infer information not explicitly visible. "
                            "If the answer cannot be determined from the image, clearly state that the information is not visible in the image."
                        )
                    if "camera_prompts" not in loaded_prompts or "default" not in loaded_prompts.get("camera_prompts", {}):
                        if "camera_prompts" not in loaded_prompts:
                            loaded_prompts["camera_prompts"] = {}
                        loaded_prompts["camera_prompts"]["default"] = (
                            "Based on the provided image, please answer the following question: "
                        )
                    return loaded_prompts
            else:
                logger.warning(f"Prompts file not found: {prompts_file}, using default prompts")
                # Default prompts with enhanced guidance
                return {
                    "system_context": (
                        "You are a precise visual analysis assistant. "
                        "Your task is to analyze the provided security camera image and answer the user's question. "
                        "Base your answer *only* on the visual information present in the image. "
                        "Be concise and direct. Do not speculate or infer information not explicitly visible. "
                        "If the answer cannot be determined from the image, clearly state that the information is not visible in the image."
                    ),
                    "camera_prompts": {
                        "default": (
                            "Based on the provided image, please answer the following question: "
                        )
                    }
                }
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            # Fallback default prompts in case of error
            return {
                "system_context": (
                    "You are a precise visual analysis assistant. "
                    "Your task is to analyze the provided security camera image and answer the user's question. "
                    "Base your answer *only* on the visual information present in the image. "
                    "Be concise and direct. Do not speculate or infer information not explicitly visible. "
                    "If the answer cannot be determined from the image, clearly state that the information is not visible in the image."
                ),
                "camera_prompts": {
                    "default": (
                        "Based on the provided image, please answer the following question: "
                    )
                }
            }
    
    def process(self):
        """Start processing the camera feed"""
        if self.running:
            return
        
        # Reset the last process time to now to prevent immediate reconnection
        self.last_process_time = time.time()
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
        
        # Start the monitoring thread with a slight delay
        self.monitoring_thread = threading.Thread(target=self._monitor_connection)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.is_active = True
        logger.info(f"OllamaVision processor started for camera {self.camera_data.get('name')}")
    
    def _processing_loop(self):
        """Main processing loop to keep the camera connection alive"""
        try:
            consecutive_errors = 0
            max_consecutive_errors = 10
            
            while self.running:
                try:
                    # Get a frame to keep the connection alive
                    frame = self._get_frame()
                    
                    if frame is not None:
                        self.last_frame = frame
                        self.last_process_time = time.time()
                        consecutive_errors = 0  # Reset error counter on success
                    else:
                        consecutive_errors += 1
                        if consecutive_errors % 10 == 0:  # Log only occasionally to avoid spam
                            logger.warning(f"Failed to get frame {consecutive_errors} times in a row")
                        
                        if consecutive_errors >= max_consecutive_errors:
                            logger.warning(f"Too many consecutive frame errors ({consecutive_errors}), will let monitor thread handle reconnection")
                            # Don't try to reconnect here, let the monitor thread handle it
                            consecutive_errors = 0  # Reset to avoid repeated warnings
                    
                    # Sleep to maintain target FPS
                    time.sleep(self.frame_interval)
                    
                except Exception as loop_error:
                    consecutive_errors += 1
                    logger.error(f"Error in frame processing: {str(loop_error)}")
                    time.sleep(1)  # Sleep longer on error
        except Exception as e:
            logger.error(f"Fatal error in processing loop: {str(e)}")
            self.running = False
    
    def _monitor_connection(self):
        """Monitor the camera connection and reconnect if necessary"""
        try:
            # Wait a bit before starting the monitoring to allow initial connection
            time.sleep(10)
            
            reconnect_in_progress = False
            last_reconnect_time = 0
            min_reconnect_interval = 30  # Minimum seconds between reconnection attempts
            
            while self.running:
                current_time = time.time()
                
                # Check if we haven't processed a frame in a while and we're not already reconnecting
                if (not reconnect_in_progress and 
                    current_time - self.last_process_time > self.connection_watchdog_interval and
                    current_time - last_reconnect_time > min_reconnect_interval):
                    
                    logger.warning(f"No frames processed in {self.connection_watchdog_interval} seconds, reconnecting to camera {self.camera_data.get('name')}")
                    
                    reconnect_in_progress = True
                    last_reconnect_time = current_time
                    
                    # Release the current capture if it exists
                    if self.cap is not None:
                        try:
                            self.cap.release()
                            self.cap = None
                            # Small delay to ensure resources are released
                            time.sleep(1)
                        except Exception as release_error:
                            logger.error(f"Error releasing camera: {str(release_error)}")
                    
                    # Attempt to reconnect
                    reconnect_success = self._connect_to_camera()
                    reconnect_in_progress = False
                    
                    if reconnect_success:
                        logger.info(f"Successfully reconnected to camera {self.camera_data.get('name')}")
                        self.last_process_time = current_time  # Reset the timer
                    else:
                        logger.error(f"Failed to reconnect to camera {self.camera_data.get('name')}")
                
                # Sleep for a bit before checking again
                time.sleep(5)
        except Exception as e:
            logger.error(f"Error in connection monitor: {str(e)}")
    
    def stop(self):
        """Stop the processor"""
        self.running = False
        self.is_active = False
        
        # Release the camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        logger.info(f"OllamaVision processor stopped for camera {self.camera_data.get('name')}")
    
    def process_query(self, query: str) -> Dict:
        """
        Process a contextual query using the Gemini vision model
        
        Args:
            query: The user's question about the camera feed
            
        Returns:
            Dict containing the response and metadata
        """
        if not self.is_active:
            return {
                "success": False,
                "error": f"Vision processor for camera {self.camera_data.get('id')} is not active. Please wait for system to initialize."
            }
        
        with self.query_lock:
            # Check if we need to wait before processing another query
            current_time = time.time()
            if current_time - self.last_query_time < self.query_cooldown:
                time.sleep(self.query_cooldown)
            
            self.last_query_time = time.time()
            
            try:
                # Get the current frame
                frame = self.last_frame
                if frame is None:
                    return {
                        "success": False,
                        "error": "No camera frame available. Please try again later."
                    }
                
                # Convert frame to base64 for the API
                success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not success:
                    return {
                        "success": False,
                        "error": "Failed to encode camera frame."
                    }
                
                image_bytes = buffer.tobytes()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                
                # Get the system context
                system_context = self.prompts.get("system_context", "")
                
                # Construct the prompt using the system context and user's query
                camera_prompt = self.prompts.get("camera_prompts", {}).get("default", "")
                full_prompt = f"{system_context}\n\n{camera_prompt}\n\nUser question: {query}"
                
                # Log the query
                logger.info(f"Processing query with {self.model}: {query}")
                logger.info(f"Calling Gemini vision API with model {self.model}, image size: {len(image_bytes) // 1024}KB")
                
                # Prepare the API request for Gemini
                payload = {
                    "contents": [{
                        "parts": [
                            {"text": full_prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_base64
                                }
                            }
                        ]
                    }]
                }
                
                # Set request headers
                headers = {
                    "Content-Type": "application/json"
                }
                
                # Make the API request to Gemini
                response = requests.post(
                    self.gemini_url,
                    headers=headers,
                    json=payload,
                    timeout=self.query_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract text from Gemini response structure
                    response_text = ""
                    try:
                        for candidate in result.get("candidates", []):
                            for part in candidate.get("content", {}).get("parts", []):
                                if "text" in part:
                                    response_text += part["text"]
                    except Exception as e:
                        logger.error(f"Error parsing Gemini response: {str(e)}")
                        return {
                            "success": False,
                            "error": f"Error parsing Gemini response: {str(e)}"
                        }
                    
                    # Log a preview of the response
                    preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
                    logger.info(f"Gemini vision API returned successful response: {preview}")
                    
                    # Save the query and response
                    self._save_query_result(query, response_text, frame)
                    
                    return {
                        "success": True,
                        "response": response_text,
                        "camera_id": self.camera_data.get("id"),
                        "camera_name": self.camera_data.get("name"),
                        "timestamp": datetime.now().isoformat(),
                        "model": self.model
                    }
                else:
                    error_msg = f"Gemini API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg
                    }
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Query timed out: {query}")
                return {
                    "success": False,
                    "error": f"Query timed out after {self.query_timeout} seconds. Please try again."
                }
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                return {
                    "success": False,
                    "error": f"Error processing query: {str(e)}"
                }
    
    def _process_frame(self, frame):
        """
        Process a frame - required implementation of abstract method from BaseProcessor
        For OllamaVision, we just store the frame for later use in queries
        """
        # We don't need to do any processing on each frame
        # Just store the latest frame for when a query comes in
        self.last_frame = frame
        return None  # No specific result to return
    
    def _save_query_result(self, query: str, response: str, frame: np.ndarray):
        """Save the query, response, and frame for later analysis"""
        try:
            # Create a unique filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{timestamp}_{self.camera_data.get('id')}"
            
            # Save the frame
            frame_path = os.path.join(self.query_output_dir, f"{filename_base}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Save the query and response
            metadata = {
                "query": query,
                "response": response,
                "camera_id": self.camera_data.get("id"),
                "camera_name": self.camera_data.get("name"),
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "frame_path": frame_path
            }
            
            metadata_path = os.path.join(self.query_output_dir, f"{filename_base}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving query result: {str(e)}")