import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

from app.core.config import settings
from app.services.camera_service import camera_service
from app.services.user_service import UserService

# Set up logging
logger = logging.getLogger(__name__)

class AIEngine:
    """
    The main AI Engine class that orchestrates rule processing and camera streams
    """
    def __init__(self):
        self.rules = {}  # Store rules by ID
        self.cameras = {}  # Store camera info by ID
        self.users = {}  # Store user info by ID
        self.active_processors = {}  # Store active processors by camera ID
        self.running = False
        self.user_service = UserService()  # Create our own instance
        self._load_data()
        self.attendance_records = {}  # Format: {date: {user_id: {'entry': time, 'exit': time, 'present': bool}}}
        self.unauthorized_logs = []  # List of unauthorized entry logs
        
        # Create logs directory if it doesn't exist
        self.logs_dir = settings.BASE_DIR / "logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Create attendance directory if it doesn't exist
        self.attendance_dir = settings.BASE_DIR / "data" / "attendance"
        os.makedirs(self.attendance_dir, exist_ok=True)
        
        # Create unauthorized directory if it doesn't exist
        self.unauthorized_dir = settings.BASE_DIR / "data" / "unauthorized"
        os.makedirs(self.unauthorized_dir, exist_ok=True)
    
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
                self.cameras = json.load(f)
                logger.info(f"Loaded {len(self.cameras)} cameras from {cameras_file}")
        
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
        """Monitor loop to process rules and camera streams"""
        try:
            while self.running:
                self._process_rules()
                self._initialize_filter_processors()  # Initialize processors for enabled filters
                time.sleep(30)  # Check every 30 seconds
        except Exception as e:
            logger.error(f"Error in monitor loop: {str(e)}")
            self.running = False
    
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
                    from .processors.ollama_vision_processor import OllamaVisionProcessor
                    
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