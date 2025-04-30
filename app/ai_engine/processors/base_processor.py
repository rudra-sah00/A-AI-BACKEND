import cv2
import logging
import os
import time
import warnings
import subprocess
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
import queue

import numpy as np

# Configure logging to suppress FFmpeg errors
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "panic"  # Suppress FFmpeg messages
logging.getLogger("libav").setLevel(logging.ERROR)  # Suppress libav messages

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """
    Base class for all rule processors
    """
    def __init__(self, rule_data: Dict, camera_data: Dict, users_data: Dict, output_dir: str):
        self.rule_data = rule_data
        self.camera_data = camera_data
        self.users_data = users_data
        self.output_dir = output_dir
        self.running = False
        self.last_process_time = 0
        
        # Performance optimization settings
        self.target_fps = 30  # Full target FPS to utilize available resources
        self.frame_interval = 1.0 / self.target_fps  # Time between frames
        self.skip_frames = 1  # Process every frame for highest quality
        self.frame_count = 0
        self.processed_frame_count = 0  # Track processed frames separately
        
        # Performance monitoring variables
        self.fps_count = 0  # Counter for FPS calculation
        self.fps_timer = time.time()  # Timer for FPS calculation
        self.processing_times = []  # Track recent processing times
        self.max_processing_times = 100  # Maximum number of times to track
        
        # Connection and frame handling
        self.cap = None
        self.last_frame = None
        self.process_lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=5)  # Reduced buffer from 10 to 5 frames
        self.processing_thread = None
        self.monitoring_thread = None
        self.last_error_time = 0  # Track when the last error occurred
        self.error_threshold = 5  # Minimum seconds between error logs
        
        # RTSP connection parameters
        self.rtsp_transport = "tcp"  # Use TCP for more reliable connection
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
        self.connection_watchdog_interval = 15  # Check connection every 15 seconds
        
        # Additional RTSP options for better stream stability
        self.rtsp_options = {
            'rtsp_transport': 'tcp',         # Use TCP (more reliable than UDP)
            'buffer_size': '10485760',       # 10MB buffer
            'max_delay': '500000',           # 500ms max delay
            'stimeout': '5000000',           # 5s timeout
            'reconnect': '1',                # Enable auto reconnect
            'reconnect_delay_max': '5',      # Max 5s between reconnect attempts
            'reconnect_streamed': '1',       # Reconnect if we get data
            'reconnect_on_network_error': '1'# Reconnect on network errors
        }
        
        # Error tracking
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10
        self.error_reset_time = time.time()
        
        # Frame optimization - DISABLED automatic downscaling
        self.downscale_factor = 1.0  # Always use full resolution
        self.max_width = 99999  # Effectively disable width constraint
        self.max_height = 99999  # Effectively disable height constraint
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def _build_rtsp_url(self, url):
        """Build RTSP URL with options for better stream handling"""
        if not url.startswith('rtsp://'):
            return url
            
        # Add options as URL parameters
        options = '&'.join([f'{k}={v}' for k, v in self.rtsp_options.items()])
        if '?' in url:
            return f"{url}&{options}"
        else:
            return f"{url}?{options}"
    
    def _connect_to_camera(self):
        """Connect to the camera stream with improved error handling"""
        try:
            rtsp_url = self.camera_data.get("rtsp_url")
            if not rtsp_url:
                logger.error(f"No RTSP URL found for camera {self.camera_data.get('id')}")
                return False
            
            # Add RTSP options to the URL for better stream stability
            rtsp_url_with_options = self._build_rtsp_url(rtsp_url)
                
            # Close existing capture if any
            if self.cap is not None:
                self.cap.release()
                
            # Connect to the RTSP stream with advanced configuration
            self.cap = cv2.VideoCapture(rtsp_url_with_options, cv2.CAP_FFMPEG)
            
            # Set additional OpenCV parameters for better stream handling
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            if not self.cap.isOpened():
                self.reconnect_attempts += 1
                logger.error(f"Failed to connect to RTSP stream: {rtsp_url} (Attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts reached for camera {self.camera_data.get('name')}")
                    return False
                    
                time.sleep(self.reconnect_delay)
                return self._connect_to_camera()  # Recursive retry
                
            # Reset reconnect counter on successful connection
            self.reconnect_attempts = 0
            self.consecutive_errors = 0
                
            # Get camera FPS from the stream itself
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0 and fps < 100:  # Sanity check - FPS should be reasonable
                # Use full FPS as reported by the camera
                self.target_fps = fps  # Don't cap at 20 FPS anymore
                self.frame_interval = 1.0 / self.target_fps
                logger.info(f"Camera {self.camera_data.get('name')} reports {fps:.2f} FPS, using full rate")
            else:
                logger.info(f"Camera {self.camera_data.get('name')} reports invalid FPS: {fps}, using default {self.target_fps} FPS")
            
            # Get frame dimensions but don't downscale regardless of size
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(f"Camera resolution: {width}x{height} - using full resolution")
            
            # Always use full resolution
            self.downscale_factor = 1.0
                
            logger.info(f"Successfully connected to camera: {self.camera_data.get('name')}")
            return True
        
        except Exception as e:
            logger.error(f"Error connecting to camera: {str(e)}")
            self.reconnect_attempts += 1
            
            if self.reconnect_attempts < self.max_reconnect_attempts:
                logger.info(f"Retrying connection in {self.reconnect_delay} seconds...")
                time.sleep(self.reconnect_delay)
                return self._connect_to_camera()  # Recursive retry
                
            return False
    
    def _get_frame(self):
        """Get a frame from the camera stream with improved error handling and quality enhancements"""
        if self.cap is None or not self.cap.isOpened():
            if not self._connect_to_camera():
                return None
        
        try:
            # Use a try-with-timeout approach to avoid hanging on stream errors
            frame = None
            
            # Read a frame with error tracking
            ret, frame = self.cap.read()
            
            # Check if the frame is valid
            if not ret or frame is None or frame.size == 0:
                self.consecutive_errors += 1
                
                # Only log occasional errors to avoid log flooding
                if self.consecutive_errors == 1 or self.consecutive_errors % 100 == 0:
                    logger.warning(f"Failed to read valid frame from camera {self.camera_data.get('name')} (error #{self.consecutive_errors})")
                
                # If we've had too many consecutive errors, try reconnecting
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.warning(f"Too many consecutive errors ({self.consecutive_errors}), reconnecting to camera {self.camera_data.get('name')}")
                    if self._connect_to_camera():
                        ret, frame = self.cap.read()
                        if not ret or frame is None or frame.size == 0:
                            return None
                    else:
                        return None
                else:
                    # Skip this frame but don't try reconnecting yet
                    return None
                    
            # Reset error counter if we successfully got a frame
            if self.consecutive_errors > 0:
                self.consecutive_errors = 0
                
            self.frame_count += 1
            self.fps_count += 1
            
            # Apply frame quality enhancement
            frame = self._enhance_frame_quality(frame)
            
            # Never apply downscaling - always use full resolution
            # The downscale_factor should always be 1.0
            
            self.last_frame = frame
            return frame
            
        except Exception as e:
            current_time = time.time()
            self.consecutive_errors += 1
            
            # Limit error logging to avoid spamming
            if current_time - self.last_error_time > self.error_threshold:
                logger.error(f"Error reading frame: {str(e)}")
                self.last_error_time = current_time
            
            # If errors persist, try reconnecting
            if self.consecutive_errors >= self.max_consecutive_errors:
                if self._connect_to_camera():
                    self.consecutive_errors = 0
                    return self._get_frame()  # Try again after reconnection
            
            return None
    
    def _enhance_frame_quality(self, frame):
        """
        Enhance frame quality by adjusting brightness, contrast, and sharpness.
        Also fixes issues with partial blurriness by applying appropriate filters.
        """
        if frame is None or frame.size == 0:
            return frame
            
        try:
            # Make a copy to avoid modifying the original
            enhanced = frame.copy()
            
            # Check if the frame has areas that are blurry (usually top or bottom half)
            # This can happen due to interlacing issues or partial frame capture
            
            # 1. Fix interlacing artifacts if present
            # This addresses issues where alternate rows may be from different captures
            height, width = enhanced.shape[:2]
            
            # Check for interlacing artifacts by comparing adjacent rows
            rows_to_check = min(height, 100)  # Check up to 100 rows
            row_diffs = []
            
            for i in range(1, rows_to_check):
                # Compare row i with row i-1
                diff = np.sum(np.abs(enhanced[i].astype(np.int32) - enhanced[i-1].astype(np.int32)))
                row_diffs.append(diff / width)  # Normalize by width
            
            # Calculate variance of differences between adjacent rows
            if len(row_diffs) > 0:
                avg_diff = sum(row_diffs) / len(row_diffs)
                variance = sum((d - avg_diff) ** 2 for d in row_diffs) / len(row_diffs)
                
                # If variance is high, it suggests interlacing or partial frame capture
                if variance > 10000:  # Threshold determined empirically
                    logger.debug(f"Detected possible interlacing artifacts, applying deinterlacing")
                    
                    # Apply deinterlacing by blending adjacent rows
                    deinterlaced = np.zeros_like(enhanced)
                    
                    # Copy first and last rows as-is
                    deinterlaced[0] = enhanced[0]
                    deinterlaced[height-1] = enhanced[height-1]
                    
                    # Blend other rows
                    for i in range(1, height-1):
                        # Weighted average of current row with neighbors
                        deinterlaced[i] = (enhanced[i-1].astype(np.float32) * 0.25 + 
                                         enhanced[i].astype(np.float32) * 0.5 + 
                                         enhanced[i+1].astype(np.float32) * 0.25).astype(np.uint8)
                    
                    enhanced = deinterlaced
            
            # 2. Apply general image enhancements for better quality
            # Adjust brightness and contrast
            alpha = 1.05  # Slight contrast increase (1.0 is no change)
            beta = 3      # Slight brightness increase 
            enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
            
            # Apply gentle sharpening if image appears soft
            # Calculate overall image sharpness
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # If image is softer than threshold, apply sharpening
            if laplacian_var < 100:  # Empirical threshold for softness
                kernel = np.array([[-1,-1,-1], 
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing frame quality: {str(e)}")
            # If enhancement fails, return original frame
            return frame
    
    def start_processing(self):
        """Start continuous frame processing in a separate thread with monitoring"""
        if self.running:
            return
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._frame_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start connection monitoring thread
        self.monitoring_thread = threading.Thread(target=self._connection_monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info(f"Started continuous processing for camera {self.camera_data.get('name')}")
    
    def stop_processing(self):
        """Stop continuous frame processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
            
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
            self.monitoring_thread = None
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def _connection_monitoring_loop(self):
        """Monitor connection and reconnect if needed"""
        last_check_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check connection periodically
                if current_time - last_check_time > self.connection_watchdog_interval:
                    if self.cap is None or not self.cap.isOpened() or self.consecutive_errors > self.max_consecutive_errors / 2:
                        logger.warning(f"Connection watchdog detected issues with camera {self.camera_data.get('name')}, reconnecting...")
                        self._connect_to_camera()
                    
                    # Log performance statistics
                    fps = self.fps_count / (current_time - last_check_time)
                    logger.info(f"Camera {self.camera_data.get('name')} - Current FPS: {fps:.2f}, " +
                              f"Frames: {self.frame_count}, Processed: {self.processed_frame_count}")
                    
                    # Calculate average processing time but don't adjust frame skipping
                    if self.processing_times:
                        avg_time = sum(self.processing_times) / len(self.processing_times)
                        logger.debug(f"Average frame processing time: {avg_time*1000:.2f}ms")
                        
                        # Don't auto-adjust skip_frames to maintain full quality
                    
                    # Reset monitoring counters
                    self.fps_count = 0
                    self.processing_times = []
                    last_check_time = current_time
                
                # Sleep to avoid consuming CPU
                time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Error in connection monitoring: {str(e)}")
                time.sleep(5.0)
    
    def _frame_processing_loop(self):
        """Process frames continuously at the target FPS rate with better error handling and frame skipping"""
        last_frame_time = time.time()
        self.fps_timer = time.time()
        self.fps_count = 0
        
        while self.running:
            try:
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                # Calculate FPS every 5 seconds
                if current_time - self.fps_timer >= 5.0:
                    if self.fps_count > 0:
                        fps = self.fps_count / (current_time - self.fps_timer)
                        logger.debug(f"Camera {self.camera_data.get('name')} processing at {fps:.2f} FPS")
                    else:
                        logger.info(f"Camera {self.camera_data.get('name')} - no frames processed in the last 5 seconds")
                    
                    self.fps_timer = current_time
                    self.fps_count = 0
                
                # Get and process a new frame at target rate
                if elapsed >= self.frame_interval:
                    frame = self._get_frame()
                    if frame is not None:
                        # Only process every Nth frame
                        if self.frame_count % self.skip_frames == 0:
                            start_time = time.time()
                            
                            # Process the frame with thread safety
                            with self.process_lock:
                                self._process_frame(frame)
                            
                            # Track processing performance
                            processing_time = time.time() - start_time
                            self.processing_times.append(processing_time)
                            if len(self.processing_times) > self.max_processing_times:
                                self.processing_times.pop(0)
                                
                            self.processed_frame_count += 1
                    
                    # Update last frame time based on actual time
                    last_frame_time = time.time()
                else:
                    # Sleep a bit to avoid tight CPU loop
                    time.sleep(max(0, min(0.01, self.frame_interval - elapsed)))
                    
            except Exception as e:
                current_time = time.time()
                # Limit error logging to avoid spamming
                if current_time - self.last_error_time > self.error_threshold:
                    logger.error(f"Error in frame processing loop: {str(e)}")
                    self.last_error_time = current_time
                
                # Don't retry too quickly on errors
                time.sleep(1.0)
    
    def process(self):
        """Process the rule by starting the continuous processing thread"""
        if not self.running:
            self.start_processing()
        return True
    
    def stop(self):
        """Stop the processor"""
        self.stop_processing()
    
    @abstractmethod
    def _process_frame(self, frame):
        """Process a frame based on the rule type"""
        pass
    
    def _save_frame(self, frame, filename):
        """Save a frame to disk with quality optimization"""
        try:
            # Use 90% JPEG quality for good file size vs quality balance
            compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
            
            # Use PNG for transparent images or images with alpha channel
            if len(frame.shape) > 2 and frame.shape[2] == 4:  # Has alpha channel
                compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # Low compression for speed
                cv2.imwrite(filename, frame, compression_params)
            else:
                cv2.imwrite(filename, frame, compression_params)
            return True
        except Exception as e:
            logger.error(f"Error saving frame: {str(e)}")
            return False