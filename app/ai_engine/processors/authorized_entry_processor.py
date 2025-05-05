from datetime import datetime, timedelta
import threading
import os
import json
import cv2
import time
import uuid
import logging
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
    logging.warning("InsightFace library not available, using alternative methods")

logger = logging.getLogger(__name__)

class AuthorizedEntryProcessor(BaseProcessor):
    """
    Processor for authorized entry rules with enhanced face recognition using InsightFace
    """
    def __init__(self, rule_data: Dict, camera_data: Dict, users_data: Dict, output_dir: str):
        super().__init__(rule_data, camera_data, users_data, output_dir)
        
        # Initialize YOLO model for person detection
        self.model = YOLOModel()
        
        # Initialize enhanced face detector with advanced capabilities
        self.face_detector = FaceDetector(min_face_size=(50, 50))  # Increase min face size for better quality
        
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
                logger.info("Initialized InsightFace for enhanced face recognition")
            except Exception as e:
                logger.warning(f"Failed to initialize InsightFace: {e}")
                self.insightface_analyzer = None
        
        # Get rule-specific data
        self.rule_condition = rule_data.get("condition", {})
        self.authorized_role = self.rule_condition.get("data", {}).get("role")
        logger.info(f"Initialized for role: {self.authorized_role}")
        
        # Unauthorized detections log
        self.unauthorized_file = os.path.join(self.output_dir, "unauthorized.json")
        self.unauthorized_logs = self._load_unauthorized_logs()
        
        # Detection settings - track quality of detected faces
        self.detection_interval = 120  # Minimum seconds between logging the same unauthorized person (increased from 60 to 120)
        self.min_quality_threshold = 0.5  # Increase minimum quality to accept a face image
        self.quality_improvement_threshold = 0.3  # Increase min improvement needed to replace an existing face
        self.person_trackers = {}  # Track detected persons across frames
        self.track_expiration_time = 90  # Seconds until a tracked person is considered "new"
        self.absence_threshold = 60  # Seconds a person must be absent to be considered "returned"
        
        # Daily person tracking - store one image per person per day
        self.daily_persons = {}  # Dictionary to track persons by day
        
        # Person absence tracking
        self.person_absence_time = {}  # Track when person was last marked as absent
        
        # Face embeddings cache for faster recognition
        self.face_embeddings_cache = {}  # Cache face embeddings for authorized users
        
        # Image directory for unauthorized detections
        self.unauthorized_images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.unauthorized_images_dir, exist_ok=True)
        
        # Create separate face directory for enhanced face images
        self.faces_dir = os.path.join(self.unauthorized_images_dir, "faces")
        os.makedirs(self.faces_dir, exist_ok=True)
        
        # Create debug directory for troubleshooting face detection issues
        self.debug_dir = os.path.join(self.unauthorized_images_dir, "debug")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Debug mode for saving additional detection info - set to False to stop saving debug images
        self.debug_mode = False  # Changed from True to False to stop filling storage with debug images
        
        # Front face validation settings
        self.min_face_angle = 30  # Maximum allowed face angle in degrees from frontal position
        self.eye_detection_required = True  # Require eye detection for validating front-facing faces
        
        # Add interactive mode flag to control whether to prompt for continuing iteration
        self.interactive_mode = True  # Set to True to enable interactive prompts
        
        # Keep track of last presence state for each unauthorized person
        self.person_presence_state = {}  # To track if a person has left and returned
        
        # InsightFace similarity threshold (adjust based on testing)
        self.insightface_similarity_threshold = 0.45  # Lower is stricter
        
        logger.info(f"Initialized AuthorizedEntryProcessor for role: {self.authorized_role} with InsightFace: {HAVE_INSIGHTFACE}")
    
    def _load_unauthorized_logs(self):
        """Load unauthorized logs from file"""
        if os.path.exists(self.unauthorized_file):
            try:
                with open(self.unauthorized_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error loading unauthorized logs from {self.unauthorized_file}")
        
        # Initialize empty list if file doesn't exist or is invalid
        return []
    
    def _save_unauthorized_logs(self):
        """Save unauthorized logs to file"""
        try:
            with open(self.unauthorized_file, "w") as f:
                json.dump(self.unauthorized_logs, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving unauthorized logs: {str(e)}")
    
    def _load_reference_images(self) -> Dict:
        """Load all reference images for authorized users with improved face extraction using InsightFace"""
        reference_images = {}
        
        for username, user_data in self.users_data.items():
            # Only load for users with the authorized role
            if self.authorized_role and user_data.get("role").lower() != self.authorized_role.lower():
                continue
                
            photo_path = user_data.get("photo_path")
            if not photo_path:
                continue
                
            # Convert relative path to absolute path if needed
            if not os.path.isabs(photo_path):
                from app.core.config import settings
                photo_path = os.path.join(settings.BASE_DIR, photo_path)
                
            # Load image if exists
            if os.path.exists(photo_path):
                try:
                    image = cv2.imread(photo_path)
                    if image is None:
                        logger.error(f"Failed to load image for user {username}: {photo_path}")
                        continue
                    
                    # Try InsightFace first for better face extraction
                    face_embedding = None
                    if HAVE_INSIGHTFACE and self.insightface_analyzer:
                        faces = self.insightface_analyzer.get(image)
                        if len(faces) > 0:
                            # Get the most confident face
                            best_face = max(faces, key=lambda x: x.det_score)
                            face_img = image[int(best_face.bbox[1]):int(best_face.bbox[3]), 
                                          int(best_face.bbox[0]):int(best_face.bbox[2])].copy()
                            face_embedding = best_face.embedding
                            
                            reference_images[username] = {
                                "image": face_img,
                                "user_data": user_data,
                                "quality": 0.9,  # InsightFace detections are usually high quality
                                "embedding": face_embedding
                            }
                            logger.info(f"Loaded InsightFace reference for {username}")
                            
                            # Cache the face embedding for faster comparison
                            self.face_embeddings_cache[username] = face_embedding
                            continue
                    
                    # Fall back to our regular face detector
                    faces = self.face_detector.detect_faces(image)
                    
                    if faces:
                        # Find highest quality face if multiple faces detected
                        best_face, best_rect, quality = self.face_detector.get_best_face(image, faces)
                        
                        if best_face is not None:
                            # Store enhanced face for better matching
                            reference_images[username] = {
                                "image": best_face,
                                "user_data": user_data,
                                "quality": quality
                            }
                            logger.info(f"Loaded reference face for {username} with quality {quality:.2f}")
                        else:
                            # If no good face detected, use the whole image
                            reference_images[username] = {
                                "image": image,
                                "user_data": user_data,
                                "quality": 0.0
                            }
                            logger.warning(f"No good face detected for {username}, using full image")
                    else:
                        # If no face detected, use the whole image
                        reference_images[username] = {
                            "image": image,
                            "user_data": user_data,
                            "quality": 0.0
                        }
                        logger.warning(f"No face detected for {username}, using full image")
                except Exception as e:
                    logger.error(f"Error loading reference image for {username}: {str(e)}")
        
        logger.info(f"Loaded {len(reference_images)} reference images for role: {self.authorized_role}")
        return reference_images
    
    def _is_same_person(self, bbox1, bbox2, frame_width, frame_height, threshold=0.5):
        """
        Determine if two bounding boxes likely belong to the same person
        Uses intersection over union (IoU) and relative positions
        """
        # Calculate IoU
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        
        x2_1 = x1_1 + w1
        y2_1 = y1_1 + h1
        x2_2 = x1_2 + w2
        y2_2 = y1_2 + h2
        
        # Area of intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return False  # No overlap
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Area of both boxes
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        # Also consider the relative positions - people don't move too far between frames
        distance = np.sqrt(((x1_1 + w1/2) - (x1_2 + w2/2))**2 + ((y1_1 + h1/2) - (y1_2 + h2/2))**2)
        distance_normalized = distance / np.sqrt(frame_width**2 + frame_height**2)
        
        # Consider it the same person if IoU is high or distance is small
        return iou > threshold or distance_normalized < 0.1
    
    def _get_today_date_str(self) -> str:
        """Get today's date as a string"""
        return datetime.now().strftime("%Y-%m-%d")
    
    def _cleanup_expired_trackers(self):
        """Remove expired person trackers with improved absence tracking"""
        current_time = time.time()
        expired_ids = []
        absent_ids = []
        
        for person_id, tracker_data in self.person_trackers.items():
            last_seen = tracker_data.get("last_seen", 0)
            
            # Check if the person hasn't been seen recently but not long enough to expire
            if current_time - last_seen > self.absence_threshold and current_time - last_seen <= self.track_expiration_time:
                # Mark as absent but don't expire yet
                if self.person_presence_state.get(person_id, True):
                    absent_ids.append(person_id)
                    self.person_presence_state[person_id] = False
                    self.person_absence_time[person_id] = current_time
            
            # If person hasn't been seen for track_expiration_time, expire them
            if current_time - last_seen > self.track_expiration_time:
                expired_ids.append(person_id)
                # If person expires, mark them as having left the scene
                self.person_presence_state[person_id] = False
        
        # Remove expired trackers
        for person_id in expired_ids:
            del self.person_trackers[person_id]
            if person_id in self.person_absence_time:
                del self.person_absence_time[person_id]
        
        if expired_ids:
            logger.debug(f"Cleared {len(expired_ids)} expired person trackers")
        if absent_ids:
            logger.debug(f"Marked {len(absent_ids)} persons as absent")
    
    def _get_or_create_tracker(self, frame, bbox, unauthorized=True):
        """
        Get an existing tracker for a person or create a new one with improved absence detection
        Returns the tracker ID
        """
        frame_height, frame_width = frame.shape[:2]
        current_time = time.time()
        
        # Check if this person matches any existing tracker
        for person_id, tracker_data in self.person_trackers.items():
            last_bbox = tracker_data.get("bbox")
            if last_bbox and self._is_same_person(bbox, last_bbox, frame_width, frame_height):
                # Update the tracker
                self.person_trackers[person_id]["bbox"] = bbox
                self.person_trackers[person_id]["last_seen"] = current_time
                
                # If this is an unauthorized person who was previously marked as having left
                if unauthorized and person_id in self.person_presence_state and not self.person_presence_state[person_id]:
                    # Check if they've been absent long enough to be considered "returned"
                    last_absent_time = self.person_absence_time.get(person_id, 0)
                    if current_time - last_absent_time >= self.absence_threshold:
                        logger.debug(f"Person {person_id} has returned after {int(current_time - last_absent_time)}s absence")
                        self.person_presence_state[person_id] = True
                        # Reset last logged time so we can capture a new photo
                        self.person_trackers[person_id]["last_logged"] = 0
                    else:
                        logger.debug(f"Person {person_id} was briefly absent for {int(current_time - last_absent_time)}s, not considered returned")
                
                # If person was present before, ensure they're marked as present now
                self.person_presence_state[person_id] = True
                
                return person_id
        
        # If no match found, create a new tracker
        person_id = str(uuid.uuid4())
        self.person_trackers[person_id] = {
            "id": person_id,
            "bbox": bbox,
            "first_seen": current_time,
            "last_seen": current_time,
            "best_face_quality": 0.0,
            "best_face_path": None,
            "is_unauthorized": unauthorized,
            "last_logged": 0
        }
        
        # Initialize presence state for new person
        self.person_presence_state[person_id] = True
        
        logger.debug(f"Created new tracker for {'unauthorized' if unauthorized else 'authorized'} person: {person_id}")
        return person_id
    
    def _is_front_facing_face(self, face_img, face_rect):
        """
        Determine if a face is front-facing (not a side profile or back of head)
        Returns a tuple of (is_front_facing, confidence)
        """
        # If no face detected, not a front-facing face
        if face_img is None or face_rect is None:
            return False, 0.0
            
        face_width, face_height = face_rect[2], face_rect[3]
        
        # Check face aspect ratio - front-facing faces should have width/height ratio around 0.75-0.85
        aspect_ratio = face_width / face_height if face_height > 0 else 0
        if aspect_ratio < 0.65 or aspect_ratio > 1.0:
            logger.debug(f"Face rejected: unusual aspect ratio {aspect_ratio:.2f}")
            return False, 0.0
            
        # Use the eye detector to validate face is front-facing
        if self.eye_detection_required:
            # Try to detect eyes in the face
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            eyes = self.face_detector.eye_cascade.detectMultiScale(gray_face, 1.1, 3)
            
            # If no eyes detected, likely not a front-facing face
            if len(eyes) < 1:
                logger.debug(f"Face rejected: no eyes detected")
                return False, 0.0
                
            # If we detect 2 eyes, check their relative positions
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate
                eyes = sorted(eyes, key=lambda e: e[0])
                
                # Check if eyes are roughly at the same height
                eye_y_diff = abs(eyes[0][1] - eyes[1][1])
                if eye_y_diff > face_height * 0.15:  # Eyes should be roughly level
                    logger.debug(f"Face rejected: eyes not level")
                    return False, 0.2  # Not completely rejected, but low confidence
                    
                # Check if eyes are at a reasonable distance apart
                eye_distance = eyes[1][0] - eyes[0][0]
                expected_distance = face_width * 0.4  # Eyes should be about 40% of face width apart
                if abs(eye_distance - expected_distance) > expected_distance * 0.4:
                    logger.debug(f"Face rejected: unusual eye distance")
                    return False, 0.3
                    
                # If we passed all checks, it's likely a front-facing face
                return True, 0.9
            
            # If we found only one eye, give moderate confidence
            return True, 0.7
        
        # If we're not checking eyes, do additional geometry checks
        
        # Try to detect nose
        nose = None
        if hasattr(self.face_detector, 'nose_cascade') and self.face_detector.nose_cascade:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            nose = self.face_detector.nose_cascade.detectMultiScale(gray_face, 1.1, 3)
        
        # If nose detected, check position (should be roughly in center)
        if nose and len(nose) > 0:
            nx, ny, nw, nh = nose[0]
            center_x = face_width / 2
            # Nose should be near the horizontal center of the face
            if abs(nx + nw/2 - center_x) > face_width * 0.25:
                logger.debug(f"Face rejected: nose off-center")
                return False, 0.4
                
            # Nose is centered, good sign of front-facing face
            return True, 0.8
            
        # If we can't validate with eyes or nose, give moderate confidence
        return True, 0.6
    
    def _update_tracker_with_face(self, person_id, face_img, face_path, quality):
        """Update a person tracker with face information"""
        if person_id not in self.person_trackers:
            return
            
        tracker = self.person_trackers[person_id]
        
        # Check if this is a front-facing face
        is_front_facing, front_confidence = self._is_front_facing_face(face_img, (0, 0, face_img.shape[1], face_img.shape[0]))
        
        # Adjust quality score based on front_confidence
        adjusted_quality = quality * front_confidence if is_front_facing else quality * 0.3
        
        # Get today's date
        today = self._get_today_date_str()
        
        # Check if we already have a face image for this person today
        has_face_today = False
        for log in self.unauthorized_logs:
            if log.get("date") == today and log.get("person_id") == person_id and log.get("has_face", False):
                has_face_today = True
                break
        
        # Only update if:
        # 1. This is the first face for this person today OR
        # 2. This face is significantly better than the previous best OR
        # 3. Person has returned after being absent and this is first capture since return
        is_returning = (person_id in self.person_presence_state and self.person_presence_state[person_id] and
                      person_id in self.person_absence_time and 
                      time.time() - self.person_absence_time.get(person_id, 0) >= self.absence_threshold)
                      
        if (not has_face_today or not tracker.get("best_face_path") or 
            adjusted_quality > tracker.get("best_face_quality", 0) + self.quality_improvement_threshold or
            is_returning):
            
            if not is_front_facing and tracker.get("best_face_path") and not is_returning:
                logger.debug(f"Not updating with non-front-facing face for person {person_id}")
                return
                
            logger.debug(f"Updating person {person_id} with {'front-facing' if is_front_facing else 'new'} face " +
                        f"(quality: {adjusted_quality:.2f}, previous: {tracker.get('best_face_quality', 0):.2f})" +
                        (", returning after absence" if is_returning else ""))
            
            # If a previous face image exists and we're replacing it with a better one, delete the old one
            old_face_path = tracker.get("best_face_path")
            if old_face_path and os.path.exists(old_face_path) and old_face_path != face_path:
                try:
                    os.remove(old_face_path)
                    logger.debug(f"Removed old face image: {old_face_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete old face image: {str(e)}")
            
            # Update with new best face
            tracker["best_face_quality"] = adjusted_quality
            tracker["best_face_path"] = face_path
            tracker["is_front_facing"] = is_front_facing
            
            # Update all unauthorized logs for this person to point to the new best face
            self._update_unauthorized_logs_for_person(person_id, face_path, adjusted_quality)
            
            # If person is returning, reset their absence time
            if is_returning:
                self.person_absence_time[person_id] = 0
        else:
            logger.debug(f"Not updating face for person {person_id}, already have face for today or quality not significantly better")
    
    def _update_unauthorized_logs_for_person(self, person_id, face_path, face_quality):
        """Update all unauthorized logs for a person to point to the new best face"""
        today = self._get_today_date_str()
        updated = False
        
        # Only update logs for the current day
        for log in self.unauthorized_logs:
            if log.get("person_id") == person_id and log.get("date") == today:
                log["face_path"] = face_path
                log["face_quality"] = face_quality
                log["has_face"] = face_path is not None
                updated = True
                
        if updated:
            self._save_unauthorized_logs()
            logger.debug(f"Updated unauthorized logs for person {person_id} with new best face for today")

    def _compare_with_insightface(self, face_img, reference_data):
        """Compare a face with reference data using InsightFace for better accuracy"""
        if not HAVE_INSIGHTFACE or self.insightface_analyzer is None:
            return False, 0.0
            
        try:
            # Check if we already have embedding in the reference data
            ref_embedding = reference_data.get("embedding")
            
            if ref_embedding is None:
                # Get embedding from reference image
                ref_img = reference_data["image"]
                ref_faces = self.insightface_analyzer.get(ref_img)
                
                if len(ref_faces) == 0:
                    return False, 0.0
                    
                ref_embedding = ref_faces[0].embedding
            
            # Get embedding for detected face
            faces = self.insightface_analyzer.get(face_img)
            
            if len(faces) == 0:
                return False, 0.0
                
            test_embedding = faces[0].embedding
            
            # Calculate cosine similarity
            similarity = np.dot(ref_embedding, test_embedding) / (np.linalg.norm(ref_embedding) * np.linalg.norm(test_embedding))
            
            # Consider it a match if similarity exceeds threshold
            is_match = similarity > self.insightface_similarity_threshold
            
            return is_match, similarity
            
        except Exception as e:
            logger.warning(f"InsightFace comparison error: {e}")
            return False, 0.0

    def _process_frame(self, frame):
        """Process a frame for authorized entry detection with improved face handling and InsightFace"""
        if frame is None or frame.size == 0:
            return
            
        # Get camera info
        camera_id = self.camera_data.get("id")
        camera_name = self.camera_data.get("name", "Unknown")
        frame_height, frame_width = frame.shape[:2]
        
        # Get today's date
        today = self._get_today_date_str()
        
        # Clean up expired person trackers
        self._cleanup_expired_trackers()
        
        # Initialize daily person tracking if needed
        if today not in self.daily_persons:
            self.daily_persons[today] = {}
            
            # Clean up old days (keep only the latest day)
            old_days = [day for day in self.daily_persons if day != today]
            for old_day in old_days:
                del self.daily_persons[old_day]
        
        # Load reference images for authorized users
        reference_images = self._load_reference_images()
        if not reference_images:
            logger.warning(f"No reference images found for role: {self.authorized_role}")
            # We'll still process the frame to detect unauthorized persons
        
        # Detect persons in the frame
        detections = self.model.detect_persons(frame)
        
        if not detections:
            return  # No persons detected, nothing to do
            
        # Debug: Save frame with person detections if in debug mode
        if self.debug_mode and len(detections) > 0:
            debug_frame = frame.copy()
            for detection in detections:
                bbox = detection["bbox"]
                x1, y1, x2, y2 = [int(c) for c in bbox]
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_path = os.path.join(self.debug_dir, f"person_detections_{timestamp}.jpg")
            cv2.imwrite(debug_path)
        
        # Track if any unauthorized persons are detected in this frame
        unauthorized_detected = False
        
        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            
            # Convert YOLO bbox to expected format (x1, y1, x2, y2) → (x1, y1, w, h)
            x1, y1, x2, y2 = bbox
            person_width = x2 - x1
            person_height = y2 - y1
            bbox_xywh = (int(x1), int(y1), int(person_width), int(person_height))
            
            # Skip low confidence detections
            if confidence < 0.5:
                continue
                
            # Extract person image for face detection
            person_img = frame[int(y1):int(y2), int(x1):int(x2)].copy()
            if person_img is None or person_img.size == 0:
                continue
            
            # Try to detect face first with InsightFace for better low-resolution performance
            best_face = None
            best_rect = None
            face_quality = 0.0
            is_insightface_detection = False
            
            if HAVE_INSIGHTFACE and self.insightface_analyzer:
                try:
                    insightface_results = self.insightface_analyzer.get(person_img)
                    if insightface_results and len(insightface_results) > 0:
                        # Get the highest confidence face
                        best_insightface = max(insightface_results, key=lambda x: x.det_score)
                        face_x1, face_y1, face_x2, face_y2 = map(int, best_insightface.bbox)
                        
                        # Extract face image
                        if (face_x1 < face_x2 and face_y1 < face_y2 and
                            face_x1 >= 0 and face_y1 >= 0 and
                            face_x2 <= person_img.shape[1] and face_y2 <= person_img.shape[0]):
                            
                            best_face = person_img[face_y1:face_y2, face_x1:face_x2].copy()
                            best_rect = (face_x1, face_y1, face_x2-face_x1, face_y2-face_y1)
                            face_quality = best_insightface.det_score  # Use detection score as quality
                            is_insightface_detection = True
                            
                            logger.debug(f"InsightFace detected face with quality {face_quality:.2f}")
                except Exception as e:
                    logger.warning(f"InsightFace detection error: {e}")
                    
            # If InsightFace didn't find a face or wasn't available, fall back to regular detection
            if best_face is None:
                # Try to detect face in the person image using multiple methods
                faces = self.face_detector.detect_faces(person_img)
                
                # Debug: Save face detection results
                if self.debug_mode and faces:
                    faces_drawn = self.face_detector.draw_face_rectangles(person_img, faces, color=(0, 0, 255))
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    debug_faces_path = os.path.join(self.debug_dir, f"faces_{timestamp}.jpg")
                    cv2.imwrite(debug_faces_path, faces_drawn)
                
                # Get best face with quality score
                best_face, best_rect, face_quality = self.face_detector.get_best_face(person_img, faces)
            
            # Check if this is a front-facing face
            is_front_facing = False
            front_confidence = 0.0
            
            if best_face is not None:
                # If we're using InsightFace, we trust its detection more
                if is_insightface_detection:
                    is_front_facing = True
                    front_confidence = 0.9  # InsightFace is good at finding front-facing faces
                else:
                    is_front_facing, front_confidence = self._is_front_facing_face(best_face, best_rect)
                
                face_quality = face_quality * front_confidence if not is_insightface_detection else face_quality
                
                # If not a front-facing face with good quality, try again with different detection
                if not is_front_facing or face_quality < self.min_quality_threshold:
                    if not is_insightface_detection:  # Only log this if it's not an InsightFace detection
                        logger.debug(f"Face rejected: not front-facing or low quality {face_quality:.2f}")
                        best_face, best_rect, face_quality = None, None, 0.0
            
            # If no good face found in the person image, try detecting directly in the frame
            if best_face is None or face_quality < self.min_quality_threshold:
                # Try detecting faces directly in the frame around the person region
                # Add some margin around the person
                margin_w = int(person_width * 0.2)
                margin_h = int(person_height * 0.2)
                
                # Ensure we stay within image bounds
                region_x1 = max(0, int(x1) - margin_w)
                region_y1 = max(0, int(y1) - margin_h)
                region_x2 = min(frame_width, int(x2) + margin_w)
                region_y2 = min(frame_height, int(y2) + margin_h)
                
                # Extract region and detect faces
                region = frame[region_y1:region_y2, region_x1:region_x2].copy()
                if region.size > 0:
                    # Try InsightFace first on the region
                    if HAVE_INSIGHTFACE and self.insightface_analyzer:
                        try:
                            insightface_results = self.insightface_analyzer.get(region)
                            if insightface_results and len(insightface_results) > 0:
                                # Get the highest confidence face
                                best_insightface = max(insightface_results, key=lambda x: x.det_score)
                                face_x1, face_y1, face_x2, face_y2 = map(int, best_insightface.bbox)
                                
                                # Adjust coordinates to the original frame
                                face_x1 += region_x1
                                face_y1 += region_y1
                                face_x2 += region_x1
                                face_y2 += region_y1
                                
                                # Extract face image from the original frame
                                if (face_x1 < face_x2 and face_y1 < face_y2 and
                                    face_x1 >= 0 and face_y1 >= 0 and
                                    face_x2 <= frame_width and face_y2 <= frame_height):
                                    
                                    best_face = frame[face_y1:face_y2, face_x1:face_x2].copy()
                                    best_rect = (face_x1, face_y1, face_x2-face_x1, face_y2-face_y1)
                                    face_quality = best_insightface.det_score
                                    is_front_facing = True
                                    front_confidence = 0.9
                                    is_insightface_detection = True
                        except Exception as e:
                            logger.warning(f"InsightFace region detection error: {e}")
                    
                    # Fall back to regular face detection if InsightFace didn't work
                    if best_face is None:
                        region_faces = self.face_detector.detect_faces(region)
                        if region_faces:
                            # Adjust face coordinates to the original frame
                            for i, (fx, fy, fw, fh) in enumerate(region_faces):
                                region_faces[i] = (fx + region_x1, fy + region_y1, fw, fh)
                            
                            # Try to get the best face from the region
                            best_face, best_rect, face_quality = self.face_detector.get_best_face(frame, region_faces)
                            
                            # Check if this is a front-facing face
                            if best_face is not None:
                                is_front_facing, front_confidence = self._is_front_facing_face(best_face, best_rect)
                                face_quality = face_quality * front_confidence
                                
                                # If not a front-facing face with good quality, reject it
                                if not is_front_facing or face_quality < self.min_quality_threshold:
                                    logger.debug(f"Region face rejected: not front-facing or low quality {face_quality:.2f}")
                                    best_face, best_rect, face_quality = None, None, 0.0
            
            # Try to match with authorized users if we have a face
            is_authorized = False
            matched_user = None
            
            if best_face is not None:
                # Debug: Save the detected face
                if self.debug_mode:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    detection_type = "insightface" if is_insightface_detection else "regular"
                    debug_face_path = os.path.join(self.debug_dir, 
                        f"best_face_{timestamp}_{face_quality:.2f}_{detection_type}.jpg")
                    cv2.imwrite(debug_face_path, best_face)
                
                for username, ref_data in reference_images.items():
                    ref_img = ref_data["image"]
                    user_data = ref_data["user_data"]
                    
                    # Try InsightFace comparison first for better accuracy
                    if is_insightface_detection and HAVE_INSIGHTFACE and self.insightface_analyzer:
                        is_match, similarity = self._compare_with_insightface(best_face, ref_data)
                        
                        if is_match:
                            is_authorized = True
                            matched_user = username
                            logger.info(f"Authorized entry (InsightFace): {username} at camera {camera_name} (similarity: {similarity:.3f})")
                            
                            # Track this person as authorized
                            self._get_or_create_tracker(frame, bbox_xywh, unauthorized=False)
                            break
                    
                    # If InsightFace didn't match or isn't available, try regular comparison
                    if not is_authorized:
                        # Compare faces - adjust threshold based on reference image quality
                        ref_quality = ref_data.get("quality", 0.0)
                        # Lower threshold for lower quality reference images
                        match_threshold = 0.6 if ref_quality > 0.5 else 0.5
                        
                        is_match, similarity = self.model.compare_with_reference(best_face, ref_img, threshold=match_threshold)
                        
                        if is_match:
                            is_authorized = True
                            matched_user = username
                            logger.info(f"Authorized entry: {username} detected at camera {camera_name} (similarity: {similarity:.2f})")
                            
                            # Track this person as authorized
                            self._get_or_create_tracker(frame, bbox_xywh, unauthorized=False)
                            break
            
            # If person not authorized, handle similar to existing code
            if not is_authorized:
                # Get or create tracker for this person
                person_id = self._get_or_create_tracker(frame, bbox_xywh, unauthorized=True)
                current_time = time.time()
                person_tracker = self.person_trackers[person_id]
                
                # Rest of the unauthorized person handling remains the same
                # ...existing code for unauthorized person detection...
                
                # Check if we already have an image for this person today
                person_has_face_today = False
                for log in self.unauthorized_logs:
                    if log.get("person_id") == person_id and log.get("date") == today and log.get("has_face", False):
                        person_has_face_today = True
                        break
                
                # Save face image if it's good quality and front-facing AND we don't have a face for this person today
                # or if the face is significantly better quality
                if best_face is not None and face_quality >= self.min_quality_threshold and is_front_facing:
                    if not person_has_face_today or face_quality > person_tracker.get("best_face_quality", 0) + self.quality_improvement_threshold:
                        detection_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Save enhanced face image with a unique identifier
                        # Include date in the filename to ensure one per day
                        face_filename = f"face_{person_id}_{today}_{detection_time}.jpg"
                        face_path = os.path.join(self.faces_dir, face_filename)
                        
                        # Save the face image
                        self._save_frame(best_face, face_path)
                        
                        # Update the tracker with face info
                        self._update_tracker_with_face(person_id, best_face, face_path, face_quality)
                        
                        # Track this person for today
                        self.daily_persons[today][person_id] = {
                            "face_path": face_path,
                            "face_quality": face_quality
                        }
                        
                        logger.debug(f"Saved face for unauthorized person {person_id} for today ({today}) with quality {face_quality:.2f}")
                
                # Determine if we should log this detection
                # Only log if:
                # 1. We've never logged this person before, OR 
                # 2. It's been more than detection_interval since last log AND they left and came back
                # 3. We've tracked them for at least a few seconds (to get more stable face detection)
                time_since_first_detection = current_time - person_tracker.get("first_seen", 0)
                last_logged = person_tracker.get("last_logged", 0)
                time_since_last_log = current_time - last_logged
                
                # Check if the person has left and returned (using presence state)
                person_returned = person_id in self.person_presence_state and self.person_presence_state[person_id]
                
                # Determine if we should log this detection based on our conditions
                should_log = (time_since_first_detection > 5.0 and  # Tracked for at least 5 seconds for better face
                            (last_logged == 0 or  # Never logged before
                              (time_since_last_log > self.detection_interval and  # Sufficient time passed
                              (not person_has_face_today or person_returned))))  # Don't have face for today or person returned
                
                if should_log:
                    # Get the best face we have for this person today
                    best_face_path = None
                    best_face_quality = 0.0
                    
                    # First check if this person has a face image recorded today in our daily tracking
                    if person_id in self.daily_persons.get(today, {}):
                        best_face_path = self.daily_persons[today][person_id].get("face_path")
                        best_face_quality = self.daily_persons[today][person_id].get("face_quality", 0)
                    else:
                        # If not in daily tracking, check if the tracker has a face
                        best_face_path = person_tracker.get("best_face_path")
                        best_face_quality = person_tracker.get("best_face_quality", 0)
                    
                    # Create unauthorized entry log with detection method info
                    unauthorized_log = {
                        "id": str(uuid.uuid4()),
                        "person_id": person_id,
                        "camera_id": camera_id,
                        "camera_name": camera_name,
                        "timestamp": datetime.now().isoformat(),
                        "date": today,
                        "face_path": best_face_path,
                        "face_quality": best_face_quality,
                        "confidence": float(confidence),
                        "bbox": [float(c) for c in bbox],  
                        "has_face": best_face_path is not None,
                        "is_front_facing": person_tracker.get("is_front_facing", False),
                        "detection_method": "insightface" if is_insightface_detection else "standard"
                    }
                    
                    # Add to logs and save
                    self.unauthorized_logs.append(unauthorized_log)
                    self._save_unauthorized_logs()
                    
                    # Update tracker with log time
                    person_tracker["last_logged"] = current_time
                    
                    logger.warning(f"Unauthorized entry detected at camera {camera_name}, has face: {best_face_path is not None}")
                    unauthorized_detected = True
                    
                    # Only save a context frame if needed for debugging
                    if not person_has_face_today and self.debug_mode:
                        detection_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Draw bounding boxes around all faces in the frame
                        all_faces = self.face_detector.detect_faces(frame)
                        if all_faces:
                            frame_with_faces = self.face_detector.draw_face_rectangles(frame, all_faces, color=(0, 0, 255))
                            
                            # Save context image
                            context_filename = f"context_{person_id}_{today}_{detection_time}.jpg"
                            context_path = os.path.join(self.unauthorized_images_dir, context_filename)
                            self._save_frame(frame_with_faces, context_path)
                            
                            # Update the unauthorized log with context image
                            unauthorized_log["context_image_path"] = context_path
                            self._save_unauthorized_logs()
                            
                            logger.info(f"Saved context image with {len(all_faces)} detected faces")
        
        # Return whether any unauthorized persons were detected in this frame
        return unauthorized_detected