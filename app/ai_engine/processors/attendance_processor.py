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
from app.ai_engine.utils.face_detector import FaceDetector

# Import InsightFace for enhanced low resolution face detection
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAVE_INSIGHTFACE = True
except ImportError:
    HAVE_INSIGHTFACE = False
    logging.warning("InsightFace library not available for attendance processor, using alternative methods")

logger = logging.getLogger(__name__)

class AttendanceProcessor(BaseProcessor):
    """
    Processor for attendance tracking rules with enhanced face recognition
    """
    def __init__(self, rule_data: Dict, camera_data: Dict, users_data: Dict, output_dir: str):
        super().__init__(rule_data, camera_data, users_data, output_dir)
        
        # Initialize YOLO model for person detection
        self.model = YOLOModel()
        
        # Initialize enhanced face detector for better face recognition
        self.face_detector = FaceDetector(min_face_size=(50, 50))
        
        # Initialize InsightFace for better low-resolution face recognition
        self.insightface_analyzer = None
        if HAVE_INSIGHTFACE:
            try:
                # Initialize InsightFace with appropriate models
                self.insightface_analyzer = FaceAnalysis(
                    name='buffalo_l',  # Using light model for better speed/accuracy balance
                    providers=['CPUExecutionProvider']  # Using CPU for compatibility
                )
                self.insightface_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("Initialized InsightFace for enhanced attendance face recognition")
            except Exception as e:
                logger.warning(f"Failed to initialize InsightFace for attendance: {e}")
                self.insightface_analyzer = None
        
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
        
        # Face embeddings cache for faster recognition
        self.face_embeddings_cache = {}  # Cache face embeddings for users
        
        # Set longer interval for attendance processing
        self.process_interval = 30  # Check every 30 seconds
        
        # Enhanced similarity threshold for better accuracy
        self.insightface_similarity_threshold = 0.45  # Lower is stricter
        self.traditional_similarity_threshold = 0.6  # For non-InsightFace comparisons
        
        logger.info(f"Initialized AttendanceProcessor for role: {self.target_role} with InsightFace: {HAVE_INSIGHTFACE}")
    
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
        """Load and preprocess reference image for a user with improved face detection"""
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
                image = cv2.imread(photo_path)
                if image is None:
                    logger.error(f"Failed to load image from path: {photo_path}")
                    return None
                
                # Try InsightFace first for better face extraction
                if HAVE_INSIGHTFACE and self.insightface_analyzer:
                    try:
                        faces = self.insightface_analyzer.get(image)
                        if faces and len(faces) > 0:
                            # Get the most confident face
                            best_face = max(faces, key=lambda x: x.det_score)
                            
                            # Extract face region
                            face_x1, face_y1, face_x2, face_y2 = map(int, best_face.bbox)
                            face_img = image[face_y1:face_y2, face_x1:face_x2].copy()
                            
                            # Cache the face embedding for this user
                            user_id = user_data.get("id")
                            if user_id:
                                self.face_embeddings_cache[user_id] = best_face.embedding
                            
                            return face_img
                    except Exception as e:
                        logger.warning(f"Error using InsightFace for reference image: {e}")
                
                # Fall back to regular face detector if InsightFace failed
                faces = self.face_detector.detect_faces(image)
                if faces:
                    # Find the best face
                    best_face, best_rect, quality = self.face_detector.get_best_face(image, faces)
                    if best_face is not None:
                        return best_face
                
                # If no face detected, return the whole image
                return image
                
            except Exception as e:
                logger.error(f"Error loading reference image: {str(e)}")
                
        return None

    def _compare_with_insightface(self, person_img, user_data) -> tuple:
        """
        Compare a detected person with a reference image using InsightFace
        
        Returns:
            Tuple of (is_match, similarity_score)
        """
        if not HAVE_INSIGHTFACE or self.insightface_analyzer is None:
            return False, 0.0
        
        user_id = user_data.get("id")
        if not user_id:
            return False, 0.0
        
        try:
            # Check if we have cached embedding for this user
            ref_embedding = self.face_embeddings_cache.get(user_id)
            
            # If no cached embedding, try to process reference image
            if ref_embedding is None:
                ref_img = self._load_reference_image(user_data)
                if ref_img is None:
                    return False, 0.0
                
                # Get embedding from reference image
                ref_faces = self.insightface_analyzer.get(ref_img)
                if not ref_faces or len(ref_faces) == 0:
                    return False, 0.0
                
                ref_embedding = ref_faces[0].embedding
                # Cache this embedding for future use
                self.face_embeddings_cache[user_id] = ref_embedding
            
            # Process the person image to extract face embedding
            person_faces = self.insightface_analyzer.get(person_img)
            if not person_faces or len(person_faces) == 0:
                return False, 0.0
            
            # Get embedding for detected face
            person_embedding = person_faces[0].embedding
            
            # Calculate cosine similarity between embeddings
            similarity = np.dot(ref_embedding, person_embedding) / (
                np.linalg.norm(ref_embedding) * np.linalg.norm(person_embedding)
            )
            
            # Determine if it's a match based on similarity threshold
            is_match = similarity > self.insightface_similarity_threshold
            
            return is_match, similarity
            
        except Exception as e:
            logger.error(f"Error comparing faces with InsightFace: {str(e)}")
            return False, 0.0
    
    def _process_frame(self, frame):
        """Process a frame for attendance tracking with enhanced face recognition"""
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
        
        if not detections:
            return  # No persons detected
            
        # Check each detection against user reference images
        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            
            # Skip low confidence detections
            if confidence < 0.5:
                continue
            
            # Convert YOLO bbox to expected format (x1, y1, x2, y2)
            x1, y1, x2, y2 = [int(c) for c in bbox]
                
            # Extract person image
            person_img = frame[y1:y2, x1:x2].copy()
            if person_img is None or person_img.size == 0:
                continue
            
            # Try to detect face using InsightFace first for better low-resolution performance
            is_insightface_detection = False
            person_face = None
            
            if HAVE_INSIGHTFACE and self.insightface_analyzer:
                try:
                    faces = self.insightface_analyzer.get(person_img)
                    if faces and len(faces) > 0:
                        # Get the highest confidence face
                        best_face = max(faces, key=lambda x: x.det_score)
                        face_x1, face_y1, face_x2, face_y2 = map(int, best_face.bbox)
                        
                        # Extract face image
                        if (face_x1 < face_x2 and face_y1 < face_y2 and
                            face_x1 >= 0 and face_y1 >= 0 and
                            face_x2 <= person_img.shape[1] and face_y2 <= person_img.shape[0]):
                            
                            person_face = person_img[face_y1:face_y2, face_x1:face_x2].copy()
                            is_insightface_detection = True
                except Exception as e:
                    logger.debug(f"InsightFace detection error in attendance: {e}")
            
            # If InsightFace didn't find a face, fall back to standard detection
            if person_face is None:
                faces = self.face_detector.detect_faces(person_img)
                if faces:
                    person_face, _, _ = self.face_detector.get_best_face(person_img, faces)
            
            # If we still don't have a face, use the whole person image
            if person_face is None:
                person_face = person_img
            
            # Compare with user reference images
            for username, user_data in self.users_data.items():
                # Skip users that don't match the target role
                if self.target_role and user_data.get("role") != self.target_role:
                    continue
                
                # Try to match using enhanced face recognition
                is_match = False
                similarity = 0.0
                
                # Use InsightFace if available for better accuracy
                if is_insightface_detection and HAVE_INSIGHTFACE and self.insightface_analyzer:
                    is_match, similarity = self._compare_with_insightface(person_face, user_data)
                
                # If no match with InsightFace, try traditional matching
                if not is_match:
                    # Load reference image
                    reference_img = self._load_reference_image(user_data)
                    if reference_img is None:
                        continue
                        
                    # Compare images using traditional method
                    is_match, similarity = self.model.compare_with_reference(
                        person_face, 
                        reference_img, 
                        threshold=self.traditional_similarity_threshold
                    )
                
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
                    detection_method = "InsightFace" if is_insightface_detection else "Standard"
                    
                    # Handle entry time
                    if in_entry_window and user_record["entry_time"] is None:
                        entry_time = datetime.now().strftime("%H:%M:%S")
                        user_record["entry_time"] = entry_time
                        
                        # Save detection image
                        detection_dir = os.path.join(self.output_dir, today, username)
                        os.makedirs(detection_dir, exist_ok=True)
                        filename = os.path.join(detection_dir, f"entry_{entry_time.replace(':', '-')}.jpg")
                        self._save_frame(person_face, filename)
                        
                        # Add detection method information
                        user_record["entry_detection_method"] = detection_method
                        user_record["entry_similarity"] = float(similarity)
                        
                        logger.info(f"Attendance: {username} entry recorded at {entry_time} (method: {detection_method}, similarity: {similarity:.3f})")
                        
                    # Handle exit time
                    elif (in_exit_window or after_exit) and (user_record["entry_time"] is not None):
                        exit_time = datetime.now().strftime("%H:%M:%S")
                        user_record["exit_time"] = exit_time
                        
                        # Check if user was present throughout the day
                        user_record["present"] = True
                            
                        # Save detection image
                        detection_dir = os.path.join(self.output_dir, today, username)
                        os.makedirs(detection_dir, exist_ok=True)
                        filename = os.path.join(detection_dir, f"exit_{exit_time.replace(':', '-')}.jpg")
                        self._save_frame(person_face, filename)
                        
                        # Add detection method information
                        user_record["exit_detection_method"] = detection_method
                        user_record["exit_similarity"] = float(similarity)
                        
                        logger.info(f"Attendance: {username} exit recorded at {exit_time} (method: {detection_method}, similarity: {similarity:.3f})")
                        
                    # Handle interval checks
                    elif self.interval_check and user_record["entry_time"] is not None:
                        check_time = datetime.now().strftime("%H:%M:%S")
                        
                        # Don't add too many checks, limit to one per hour
                        last_check = user_record["presence_checks"][-1] if user_record["presence_checks"] else None
                        if last_check is None or (datetime.strptime(check_time, "%H:%M:%S") - 
                                              datetime.strptime(last_check.get("time", "00:00:00"), "%H:%M:%S")).total_seconds() > 3600:
                            
                            # Add more detailed presence check information
                            check_info = {
                                "time": check_time,
                                "detection_method": detection_method,
                                "similarity": float(similarity)
                            }
                            user_record["presence_checks"].append(check_info)
                            logger.info(f"Attendance: {username} presence check at {check_time} (method: {detection_method}, similarity: {similarity:.3f})")
        
        # Save attendance records
        self._save_attendance_records()