import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
import asyncio

from app.core.config import settings
from app.services.camera_service import camera_service
from app.services.user_service import UserService
from app.utils.stream_validator import StreamValidator

# Set up logging
logger = logging.getLogger(__name__)

class AIEngine:
    """
    The main AI Engine class that orchestrates rule processing and camera streams
    """
    def __init__(self):
        self.rules: Dict[str, Any] = {}  # Store rules by ID
        self.cameras: Dict[str, Any] = {}  # Store camera info by ID
        self.users: Dict[str, Any] = {}  # Store user info by ID
        self.active_processors: Dict[str, Any] = {}  # Store active processors by camera ID
        self.running: bool = False
        self.user_service = UserService()  # Create our own instance
        self.attendance_records: Dict[str, Any] = {}  # Format: {date: {user_id: {'entry': time, 'exit': time, 'present': bool}}}
        self.unauthorized_logs: List[Any] = []  # List of unauthorized entry logs
        self.camera_active_status_cache: Dict[str, bool] = {}  # Initialize cache here

        # Create logs directory if it doesn't exist
        self.logs_dir = settings.BASE_DIR / "logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Create attendance directory if it doesn't exist
        self.attendance_dir = settings.BASE_DIR / "data" / "attendance"
        os.makedirs(self.attendance_dir, exist_ok=True)
        
        # Create unauthorized directory if it doesn't exist
        self.unauthorized_dir = settings.BASE_DIR / "data" / "unauthorized"
        os.makedirs(self.unauthorized_dir, exist_ok=True)
    
        self._load_data()  # Moved _load_data() call to the end of __init__
    
    def _load_data(self):
        """Load rules, cameras, and users data from JSON files"""
        # Load rules
        rules_file = settings.DATA_DIR / "rules.json"
        if os.path.exists(rules_file):
            with open(rules_file, 'r') as f:
                self.rules = json.load(f)
                logger.info(f"Loaded {len(self.rules)} rules from {rules_file}")
        
        # Load cameras
        cameras_file = settings.DATA_DIR / "cameras.json"
        if os.path.exists(cameras_file):
            with open(cameras_file, 'r') as f:
                loaded_cameras = json.load(f)
                for cam_id, cam_data in loaded_cameras.items():
                    self.cameras[cam_id] = cam_data  # Ensure self.cameras is populated
                    self.camera_active_status_cache[cam_id] = cam_data.get("is_active", False)
                logger.info(f"Loaded {len(self.cameras)} cameras from {cameras_file}")
        else:
            self.cameras = {}
            logger.info(f"Cameras file not found at {cameras_file}")
        
        # Load users
        users_file = settings.DATA_DIR / "users.json"
        if os.path.exists(users_file):
            with open(users_file, 'r') as f:
                self.users = json.load(f)
                logger.info(f"Loaded {len(self.users)} users from {users_file}")
    
    def start(self):
        """Start the AI engine processing"""
        if self.running:
            logger.info("AI Engine is already running")
            return
        
        self.running = True
        self._load_data()  # Reload data before starting
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("AI Engine started successfully")
    
    def stop(self):
        """Stop the AI engine processing"""
        if not self.running:
            logger.info("AI Engine is not running")
            return
        
        self.running = False
        
        # Clean up any active processors
        for camera_id, processor in self.active_processors.items():
            processor.stop()
        
        self.active_processors = {}
        logger.info("AI Engine stopped successfully")
    
    def _monitor_loop(self):
        """Monitor loop to process rules, camera streams, and update active status."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while self.running:
                # Check and update camera active statuses
                loop.run_until_complete(self._check_camera_streams_async())
                
                self._process_rules()  # This remains synchronous for now
                self._initialize_filter_processors()  # This remains synchronous
                
                time.sleep(settings.AI_ENGINE_MONITOR_INTERVAL_SECONDS)  # Use a configurable interval
        except Exception as e:
            logger.error(f"Error in monitor loop: {str(e)}", exc_info=True)
            self.running = False
        finally:
            loop.close()
    
    async def _check_camera_streams_async(self):
        """Periodically checks all camera streams and updates their active status."""
        logger.debug("Checking camera stream statuses...")
        # Create a temporary copy of camera IDs to iterate over, in case self.cameras is modified elsewhere
        current_camera_ids = list(self.cameras.keys())

        for camera_id in current_camera_ids:
            camera_data = self.cameras.get(camera_id)
            if not camera_data or not camera_data.get("rtsp_url"):
                logger.debug(f"Camera {camera_id} has no data or RTSP URL, skipping status check.")
                continue

            rtsp_url = camera_data["rtsp_url"]
            is_currently_active = False  # Assume inactive until proven active
            try:
                # Use StreamValidator.is_stream_accessible or a similar lightweight check
                # For a more robust check, you might try to grab a frame, but that's heavier.
                # StreamValidator.validate_rtsp_stream returns a dict, we need a boolean.
                # Let's assume StreamValidator has or we add a method like `is_stream_live`
                # For now, we'll simulate with a placeholder or adapt StreamValidator
                
                # Placeholder for actual stream check logic:
                # This should be a non-blocking check if possible, or run in an executor.
                # For simplicity, let's use the existing validate_rtsp_stream and extract is_valid.
                # This is a blocking call, so it's not ideal for a quick async loop.
                # Consider making StreamValidator.validate_rtsp_stream async or using a thread pool executor.
                
                # Simplified approach: Use a timeout with the validator if possible
                # For this example, we will call it directly. If it's too slow, it needs optimization.
                validation_result = await asyncio.to_thread(StreamValidator.validate_rtsp_stream, rtsp_url)
                is_currently_active = validation_result.get("is_valid", False)
                logger.debug(f"Camera {camera_id} ({rtsp_url}) validation: {validation_result}")

            except Exception as e:
                logger.error(f"Error validating stream for camera {camera_id} ({rtsp_url}): {e}")
                is_currently_active = False

            # Update status if it has changed from the cached status
            previous_status = self.camera_active_status_cache.get(camera_id, False)
            if is_currently_active != previous_status:
                logger.info(f"Camera {camera_id} status changed: {previous_status} -> {is_currently_active}")
                updated_camera = await camera_service.update_camera_active_status(camera_id, is_currently_active)
                if updated_camera:
                    self.camera_active_status_cache[camera_id] = updated_camera.is_active
                    # Update self.cameras entry as well to keep it in sync with the persisted data
                    self.cameras[camera_id]["is_active"] = updated_camera.is_active
                else:
                    logger.warning(f"Failed to update active status for camera {camera_id} in service.")
            else:
                logger.debug(f"Camera {camera_id} status unchanged ({is_currently_active}).")
    
    def _initialize_filter_processors(self):
        """Initialize processors for cameras with specific filters enabled (without rules)"""
        try:
            # Initialize any filter processors here if needed
            pass
        except Exception as e:
            logger.error(f"Error initializing filter processors: {str(e)}")
    
    def _process_rules(self):
        """Process all enabled rules"""
        try:
            # Reload the latest data
            self._load_data()
            
            # Process each rule
            for rule_id, rule_data in self.rules.items():
                if not rule_data.get("enabled", False):
                    continue
                
                camera_id = rule_data.get("cameraId")
                event_type = rule_data.get("event")
                
                # Skip if no camera ID or camera doesn't exist
                if not camera_id or camera_id not in self.cameras:
                    continue
                
                camera = self.cameras[camera_id]
                
                # Check if camera's filters have the required filter enabled
                filters_enabled = False
                for filter_config in camera.get("filters", []):
                    if filter_config.get("filter_name", "").lower() == event_type.lower() and filter_config.get("enabled", False):
                        filters_enabled = True
                        break
                
                if not filters_enabled:
                    logger.info(f"Camera {camera_id} does not have the {event_type} filter enabled, skipping rule {rule_id}")
                    continue
                
                # Process based on rule type
                if event_type == "attendance":
                    from .processors.attendance_processor import AttendanceProcessor
                    
                    # Get or create attendance processor for this rule
                    if rule_id not in self.active_processors:
                        self.active_processors[rule_id] = AttendanceProcessor(
                            rule_data, 
                            camera, 
                            self.users,
                            self.attendance_dir
                        )
                    
                    # Process the rule
                    self.active_processors[rule_id].process()
                
                elif event_type == "authorized_entry":
                    from .processors.authorized_entry_processor import AuthorizedEntryProcessor
                    
                    # Get or create authorized entry processor for this rule
                    if rule_id not in self.active_processors:
                        self.active_processors[rule_id] = AuthorizedEntryProcessor(
                            rule_data, 
                            camera, 
                            self.users,
                            self.unauthorized_dir
                        )
                    
                    # Process the rule
                    self.active_processors[rule_id].process()
                
                elif event_type.lower() == "ollamavision":
                    from .processors.ai_vision_processor import OllamaVisionProcessor
                    
                    # Get or create Ollama Vision processor for this rule
                    if rule_id not in self.active_processors:
                        self.active_processors[rule_id] = OllamaVisionProcessor(
                            rule_data,
                            camera,
                            self.users,
                            self.unauthorized_dir
                        )
                    
                    # Process the rule
                    self.active_processors[rule_id].process()
        
        except Exception as e:
            logger.error(f"Error processing rules: {str(e)}")
    
    def get_attendance_records(self, date=None):
        """Get attendance records for a specific date or all dates"""
        if not date:
            return self.attendance_records
        
        return self.attendance_records.get(date, {})
    
    def get_unauthorized_logs(self, date=None):
        """Get unauthorized entry logs for a specific date or all dates"""
        if not date:
            return self.unauthorized_logs
        
        # Filter logs for the specified date
        date_logs = [log for log in self.unauthorized_logs if log.get("date") == date]
        return date_logs