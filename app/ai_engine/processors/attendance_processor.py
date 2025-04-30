import cv2
import logging
import os
import json
import time
from datetime import datetime, timedelta
import threading
from typing import Dict, List, Any, Optional

import numpy as np

from app.ai_engine.processors.base_processor import BaseProcessor
from app.ai_engine.models.yolo_model import YOLOModel

logger = logging.getLogger(__name__)

class AttendanceProcessor(BaseProcessor):
    """
    Processor for attendance tracking rules
    """
    def __init__(self, rule_data: Dict, camera_data: Dict, users_data: Dict, output_dir: str):
        super().__init__(rule_data, camera_data, users_data, output_dir)
        
        # Initialize YOLO model
        self.model = YOLOModel()
        
        # Get rule-specific data
        self.rule_condition = rule_data.get("condition", {})
        self.target_role = self.rule_condition.get("data", {}).get("role")
        
        # Get time windows from rule
        time_data = self.rule_condition.get("data", {})
        self.entry_time_start = time_data.get("entryTimeStart", "09:00")
        self.entry_time_end = time_data.get("entryTimeEnd", "10:00")
        self.exit_time = time_data.get("exitTime", "18:00")
        self.interval_check = time_data.get("intervalCheck", False)
        
        # Attendance tracking
        self.attendance_file = os.path.join(self.output_dir, "attendance.json")
        self.attendance_records = self._load_attendance_records()
        
        # Last detection info
        self.last_detections = {}
        
        # Set longer interval for attendance processing
        self.process_interval = 30  # Check every 30 seconds
    
    def _load_attendance_records(self) -> Dict:
        """Load attendance records from file"""
        if os.path.exists(self.attendance_file):
            try:
                with open(self.attendance_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error loading attendance records from {self.attendance_file}")
        
        # Initialize empty records if file doesn't exist or is invalid
        return {}
    
    def _save_attendance_records(self):
        """Save attendance records to file"""
        try:
            with open(self.attendance_file, "w") as f:
                json.dump(self.attendance_records, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving attendance records: {str(e)}")
    
    def _is_within_entry_time(self) -> bool:
        """Check if current time is within entry time window"""
        now = datetime.now().time()
        
        # Convert entry times to datetime.time objects
        entry_start = datetime.strptime(self.entry_time_start, "%H:%M").time()
        entry_end = datetime.strptime(self.entry_time_end, "%H:%M").time()
        
        return entry_start <= now <= entry_end
    
    def _is_within_exit_time(self) -> bool:
        """Check if current time is within exit time window"""
        now = datetime.now().time()
        
        # Convert exit time to datetime.time object
        exit_time = datetime.strptime(self.exit_time, "%H:%M").time()
        
        # Check if current time is within 30 minutes of exit time
        exit_time_start = (datetime.combine(datetime.today(), exit_time) - timedelta(minutes=30)).time()
        exit_time_end = (datetime.combine(datetime.today(), exit_time) + timedelta(minutes=30)).time()
        
        return exit_time_start <= now <= exit_time_end
    
    def _is_after_exit_time(self) -> bool:
        """Check if current time is after exit time"""
        now = datetime.now().time()
        exit_time = datetime.strptime(self.exit_time, "%H:%M").time()
        return now > exit_time
    
    def _get_today_date_str(self) -> str:
        """Get today's date as a string"""
        return datetime.now().strftime("%Y-%m-%d")
    
    def _load_reference_image(self, user_data) -> Optional[np.ndarray]:
        """Load reference image for a user"""
        photo_path = user_data.get("photo_path")
        if not photo_path:
            return None
            
        # Convert the relative path to absolute path if needed
        if not os.path.isabs(photo_path):
            from app.core.config import settings
            photo_path = os.path.join(settings.BASE_DIR, photo_path)
            
        # Load the image if the file exists
        if os.path.exists(photo_path):
            try:
                return cv2.imread(photo_path)
            except Exception as e:
                logger.error(f"Error loading reference image: {str(e)}")
                
        return None
    
    def _process_frame(self, frame):
        """Process a frame for attendance tracking"""
        # Get today's date
        today = self._get_today_date_str()
        
        # Initialize attendance record for today if it doesn't exist
        if today not in self.attendance_records:
            self.attendance_records[today] = {}
            
        # Detect persons in the frame
        detections = self.model.detect_persons(frame)
        
        # Check if we are in an attendance-relevant time window
        in_entry_window = self._is_within_entry_time()
        in_exit_window = self._is_within_exit_time()
        after_exit = self._is_after_exit_time()
        
        # If we're not in a relevant time window and we don't need to do interval checks, we can skip
        if not (in_entry_window or in_exit_window or after_exit or self.interval_check):
            return
            
        # Check each detection against user reference images
        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            
            # Skip low confidence detections
            if confidence < 0.5:
                continue
                
            # Extract person image
            person_img = self.model.extract_person_image(frame, bbox)
            if person_img is None:
                continue
                
            # Compare with user reference images
            for username, user_data in self.users_data.items():
                # Skip users that don't match the target role
                if self.target_role and user_data.get("role") != self.target_role:
                    continue
                    
                # Load reference image
                reference_img = self._load_reference_image(user_data)
                if reference_img is None:
                    continue
                    
                # Compare images
                is_match, similarity = self.model.compare_with_reference(person_img, reference_img)
                
                if is_match:
                    user_id = user_data.get("id")
                    
                    # Initialize user record if not exists
                    if user_id not in self.attendance_records[today]:
                        self.attendance_records[today][user_id] = {
                            "username": username,
                            "entry_time": None,
                            "exit_time": None,
                            "presence_checks": [],
                            "present": False
                        }
                        
                    user_record = self.attendance_records[today][user_id]
                    
                    # Handle entry time
                    if in_entry_window and user_record["entry_time"] is None:
                        entry_time = datetime.now().strftime("%H:%M:%S")
                        user_record["entry_time"] = entry_time
                        
                        # Save detection image
                        detection_dir = os.path.join(self.output_dir, today, username)
                        os.makedirs(detection_dir, exist_ok=True)
                        filename = os.path.join(detection_dir, f"entry_{entry_time.replace(':', '-')}.jpg")
                        self._save_frame(person_img, filename)
                        
                        logger.info(f"Attendance: {username} entry recorded at {entry_time}")
                        
                    # Handle exit time
                    elif in_exit_window or after_exit:
                        exit_time = datetime.now().strftime("%H:%M:%S")
                        user_record["exit_time"] = exit_time
                        
                        # Check if user was present throughout the day
                        if user_record["entry_time"] is not None:
                            user_record["present"] = True
                            
                        # Save detection image
                        detection_dir = os.path.join(self.output_dir, today, username)
                        os.makedirs(detection_dir, exist_ok=True)
                        filename = os.path.join(detection_dir, f"exit_{exit_time.replace(':', '-')}.jpg")
                        self._save_frame(person_img, filename)
                        
                        logger.info(f"Attendance: {username} exit recorded at {exit_time}")
                        
                    # Handle interval checks
                    elif self.interval_check:
                        check_time = datetime.now().strftime("%H:%M:%S")
                        
                        # Don't add too many checks, limit to one per hour
                        last_check = user_record["presence_checks"][-1] if user_record["presence_checks"] else None
                        if last_check is None or (datetime.strptime(check_time, "%H:%M:%S") - 
                                               datetime.strptime(last_check, "%H:%M:%S")).total_seconds() > 3600:
                            user_record["presence_checks"].append(check_time)
                            logger.info(f"Attendance: {username} presence check at {check_time}")
        
        # Save attendance records
        self._save_attendance_records()