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

# WebsocketManager would typically be imported, e.g., from app.core.websocket_manager import WebsocketManager
# Using Any for type hinting if the exact class structure is not available here.
from app.core.config import settings # Already used

class AuthorizedEntryProcessor(BaseProcessor):
    """
    Processor for authorized entry rules with enhanced face recognition using InsightFace
    """
    def __init__(self, rule_data: Dict, camera_data: Dict, users_data: Dict, output_dir: str, websocket_manager: Optional[Any] = None): # Added websocket_manager
        super().__init__(rule_data, camera_data, users_data, output_dir)
        self.websocket_manager = websocket_manager # Store websocket_manager
        
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
        self.min_quality_threshold = 0.3  # Lowered from 0.5 to 0.3 to accept more faces
        self.quality_improvement_threshold = 0.2  # Lowered from 0.3 to 0.2 for more frequent updates
        self.person_trackers = {}  # Track detected persons across frames
        self.track_expiration_time = 90  # Seconds until a tracked person is considered "new"
        self.absence_threshold = 60  # Seconds a person must be absent to be considered "returned"
        
        # Daily person tracking - store one image per person per day
        self.daily_persons = {}  # Dictionary to track persons by day
        
        # Person absence tracking
        self.person_absence_time = {}  # Track when person was last marked as absent
        
        # Track which unauthorized persons we've already saved images for (to avoid duplicates)
        self.unauthorized_person_images = set()
        
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
        
        # Frame counter for limiting debug image saving frequency
        self.frame_counter = 0
        self.debug_save_frequency = 300  # Only save debug images every 300 frames (10 seconds at 30fps)
        
        # Front face validation settings
        self.min_face_angle = 30  # Maximum allowed face angle in degrees from frontal position
        self.eye_detection_required = True  # Require eye detection for validating front-facing faces
        
        # Add interactive mode flag to control whether to prompt for continuing iteration
        self.interactive_mode = True  # Set to True to enable interactive prompts
        
        # Keep track of last presence state for each unauthorized person
        self.person_presence_state = {}  # To track if a person has left and returned
        
        # InsightFace similarity threshold (adjust based on testing)
        self.insightface_similarity_threshold = 0.35  # Lower is stricter (was 0.45)
        
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
        
        # Static tracking to avoid spamming logs with the same messages repeatedly
        if not hasattr(self, '_reference_images_loaded'):
            self._reference_images_loaded = {}
        
        for username, user_data in self.users_data.items():
            # Only load for users with the authorized role
            if self.authorized_role and user_data.get("role").lower() != self.authorized_role.lower():
                continue
                
            # Skip if already loaded for this user
            if username in self._reference_images_loaded:
                if username in reference_images:
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
                    
                    # Save the original image for debugging
                    debug_dir = os.path.join(self.output_dir, "images", "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(debug_dir, f"original_{username}.jpg"), image)
                    
                    # Try InsightFace first for better face extraction
                    face_embedding = None
                    if HAVE_INSIGHTFACE and self.insightface_analyzer:
                        try:
                            faces = self.insightface_analyzer.get(image)
                            if len(faces) > 0:
                                # Get the most confident face
                                best_face = max(faces, key=lambda x: x.det_score)
                                
                                # Make sure bbox values are valid
                                if (best_face.bbox[0] < best_face.bbox[2] and 
                                    best_face.bbox[1] < best_face.bbox[3] and
                                    best_face.bbox[0] >= 0 and best_face.bbox[1] >= 0 and
                                    best_face.bbox[2] <= image.shape[1] and 
                                    best_face.bbox[3] <= image.shape[0]):
                                    
                                    face_img = image[int(best_face.bbox[1]):int(best_face.bbox[3]), 
                                                int(best_face.bbox[0]):int(best_face.bbox[2])].copy()
                                    face_embedding = best_face.embedding
                                    
                                    # Save detected face for debugging
                                    cv2.imwrite(os.path.join(debug_dir, f"insightface_{username}.jpg"), face_img)
                                    
                                    reference_images[username] = {
                                        "image": face_img,
                                        "user_data": user_data,
                                        "quality": 0.9,  # InsightFace detections are usually high quality
                                        "embedding": face_embedding
                                    }
                                    if username not in self._reference_images_loaded:
                                        logger.info(f"Loaded InsightFace reference for {username}")
                                        self._reference_images_loaded[username] = "insightface"
                                    
                                    # Cache the face embedding for faster comparison
                                    self.face_embeddings_cache[username] = face_embedding
                                    continue
                        except Exception as e:
                            logger.debug(f"InsightFace detection failed for {username}: {e}")
                    
                    # Try our regular face detector with very permissive parameters
                    try:
                        # Use much lower validation score for reference images
                        faces = self.face_detector.detect_faces(image, min_validation_score=0.1)
                        
                        # Save image with detected faces for debugging
                        if faces:
                            faces_drawn = self.face_detector.draw_face_rectangles(image, faces, color=(0, 255, 0))
                            cv2.imwrite(os.path.join(debug_dir, f"faces_{username}.jpg"), faces_drawn)
                        
                        if faces:
                            # Find highest quality face if multiple faces detected
                            best_face, best_rect, quality = self.face_detector.get_best_face(image, faces)
                            
                            if best_face is not None:
                                # Save detected face for debugging
                                cv2.imwrite(os.path.join(debug_dir, f"detected_{username}.jpg"), best_face)
                                
                                # Store enhanced face for better matching
                                reference_images[username] = {
                                    "image": best_face,
                                    "user_data": user_data,
                                    "quality": quality
                                }
                                if username not in self._reference_images_loaded:
                                    logger.info(f"Loaded reference face for {username} with quality {quality:.2f}")
                                    self._reference_images_loaded[username] = "standard"
                                continue
                    except Exception as e:
                        logger.debug(f"Regular face detection failed for {username}: {e}")
                    
                    # If detection methods failed, extract face using simple heuristics for photos
                    # This assumes the reference image is a portrait/ID photo
                    h, w = image.shape[:2]
                    
                    # For portrait photos, the face is typically in the upper portion
                    # Use the top 60% of the image as the face region
                    face_h = int(h * 0.6)
                    face_img = image[0:face_h, 0:w].copy()
                    
                    # Save heuristic face crop for debugging
                    cv2.imwrite(os.path.join(debug_dir, f"heuristic_{username}.jpg"), face_img)
                    
                    # Use this as a fallback with moderate quality
                    reference_images[username] = {
                        "image": face_img,  # Use cropped face image instead of full image
                        "user_data": user_data,
                        "quality": 0.7  # Increased from 0.5 to 0.7 for better matching
                    }
                    if username not in self._reference_images_loaded:
                        # Change from warning to debug to reduce log spam
                        logger.debug(f"Using heuristic face detection for {username}, using upper portion of image with good quality")
                        self._reference_images_loaded[username] = "heuristic"
                    
                except Exception as e:
                    logger.error(f"Error loading reference image for {username}: {str(e)}")
                    
                    # As a last resort, use the entire image
                    reference_images[username] = {
                        "image": image,
                        "user_data": user_data,
                        "quality": 0.5  # Moderate quality for full images
                    }
                    if username not in self._reference_images_loaded:
                        logger.warning(f"Using full image for {username} due to error: {str(e)}")
                        self._reference_images_loaded[username] = "full"
        
        # Only log if we haven't loaded them before to reduce repetitive logs
        if not hasattr(self, '_reference_images_logged'):
            logger.info(f"Loaded {len(reference_images)} reference images for role: {self.authorized_role}")
            self._reference_images_logged = True
        
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
        
        # Detect persons in the frame - without min_confidence parameter
        detections = self.model.detect_persons(frame)
        
        # Filter the detections based on confidence after the call
        min_confidence = 0.4  # Define the confidence threshold
        detections = [d for d in detections if d["confidence"] >= min_confidence]
        
        if not detections:
            return  # No persons detected, nothing to do
            
        # Turn on debug mode temporarily to help troubleshoot
        self.debug_mode = True
        
        # Debug: Save frame with person detections if in debug mode
        if self.debug_mode and len(detections) > 0:
            debug_frame = frame.copy()
            for detection in detections:
                bbox = detection["bbox"]
                x1, y1, x2, y2 = [int(c) for c in bbox]
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_path = os.path.join(self.debug_dir, f"person_detections_{timestamp}.jpg")
            cv2.imwrite(debug_path, debug_frame)
            logger.info(f"Saved debug frame with {len(detections)} person detections to {debug_path}")
        
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
            if confidence < 0.4:  # Lower threshold from 0.5 to 0.4
                continue
                
            # Extract person image for face detection
            person_img = frame[int(y1):int(y2), int(x1):int(x2)].copy()
            if person_img.size == 0 or person_img.shape[0] == 0 or person_img.shape[1] == 0:
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
                            
                            # Save debug image for troubleshooting
                            if self.debug_mode:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                debug_face_path = os.path.join(self.debug_dir, f"insightface_{timestamp}.jpg")
                                cv2.imwrite(debug_face_path, best_face)
                except Exception as e:
                    logger.warning(f"InsightFace detection error: {e}")
                    
            # If InsightFace didn't find a face or wasn't available, fall back to regular detection
            if best_face is None:
                # Try to detect face in the person image using multiple methods
                faces = self.face_detector.detect_faces(person_img, min_validation_score=0.2)  # Lower validation score from default 0.6
                
                # Debug: Save face detection results
                if self.debug_mode and faces:
                    faces_drawn = self.face_detector.draw_face_rectangles(person_img, faces, color=(0, 0, 255))
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    debug_faces_path = os.path.join(self.debug_dir, f"faces_{timestamp}.jpg")
                    cv2.imwrite(debug_faces_path, faces_drawn)
                    logger.info(f"Saved faces debug image with {len(faces)} face detections")
                
                # Get best face with quality score - set a lower threshold
                best_face, best_rect, face_quality = self.face_detector.get_best_face(person_img, faces)
                
                # Save the best face for debugging
                if self.debug_mode and best_face is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    debug_path = os.path.join(self.debug_dir, f"best_face_{timestamp}.jpg")
                    cv2.imwrite(debug_path, best_face)
                    logger.info(f"Saved best face with quality {face_quality} to {debug_path}")
            
            # If still no face found, try once more with very permissive parameters
            if best_face is None:
                # Try detecting directly in the frame with a larger region around the person
                margin_w = int(person_width * 0.4)  # Increased from 0.2 to 0.4
                margin_h = int(person_height * 0.4)  # Increased from 0.2 to 0.4
                
                # Ensure we stay within image bounds
                region_x1 = max(0, int(x1) - margin_w)
                region_y1 = max(0, int(y1) - margin_h)
                region_x2 = min(frame_width, int(x2) + margin_w)
                region_y2 = min(frame_height, int(y2) + margin_h)
                
                # Extract region and detect faces
                region = frame[region_y1:region_y2, region_x1:region_x2].copy()
                if region.size > 0:
                    # Try to detect faces with even lower threshold
                    region_faces = self.face_detector.detect_faces(region, min_validation_score=0.2)
                    
                    # Debug: Save region and faces
                    if self.debug_mode and len(region_faces) > 0:
                        region_drawn = self.face_detector.draw_face_rectangles(region, region_faces, color=(255, 0, 0))
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        debug_region_path = os.path.join(self.debug_dir, f"region_{timestamp}.jpg")
                        cv2.imwrite(debug_region_path, region_drawn)
                        logger.info(f"Saved region with {len(region_faces)} faces to {debug_region_path}")
                    
                    if region_faces:
                        # Get the best face from the region
                        region_best_face, region_best_rect, region_face_quality = self.face_detector.get_best_face(region, region_faces)
                        
                        if region_best_face is not None:
                            best_face = region_best_face
                            best_rect = region_best_rect  # This is relative to the region
                            face_quality = region_face_quality
                            
                            # Save debug image
                            if self.debug_mode:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                debug_region_best_path = os.path.join(self.debug_dir, f"region_best_{timestamp}.jpg")
                                cv2.imwrite(debug_region_best_path, best_face)
            
            # Try to match with authorized users if we have a face
            is_authorized = False
            matched_user = None
            
            if best_face is not None:
                # Try to match with each reference image
                for username, ref_data in reference_images.items():
                    ref_img = ref_data["image"]
                    user_data = ref_data["user_data"]
                    ref_quality = ref_data.get("quality", 0.0)
                    
                    # Try InsightFace comparison first for better accuracy
                    if is_insightface_detection and HAVE_INSIGHTFACE and self.insightface_analyzer:
                        is_match, similarity = self._compare_with_insightface(best_face, ref_data)
                        
                        if is_match:
                            is_authorized = True
                            matched_user = username
                            logger.info(f"Authorized entry (InsightFace): {username} at camera {camera_name} (similarity: {similarity:.3f})")
                            
                            # Track this person as authorized
                            self._get_or_create_tracker(frame, bbox_xywh, unauthorized=False)
                            
                            # Send notification to frontend
                            if self.websocket_manager:
                                notification_data = {
                                    "type": "authorized_entry",
                                    "camera_id": camera_id,
                                    "camera_name": camera_name,
                                    "user_name": matched_user,
                                    "timestamp": datetime.now().isoformat(),
                                    "message": f"Authorized entry: {matched_user} at {camera_name} (Method: InsightFace)"
                                }
                                try:
                                    self.websocket_manager.broadcast(notification_data) 
                                    logger.info(f"Sent authorized_entry notification for {matched_user} via WebSocket.")
                                except Exception as e:
                                    logger.error(f"Failed to send authorized_entry notification via WebSocket: {e}")
                            break
                    
                    # If not matched with InsightFace, use regular comparison
                    if not is_authorized:
                        # Use a lower threshold for the comparison since we're already filtering with quality score
                        match_threshold = 0.4  # Lowered from 0.5/0.6 to 0.4
                        
                        is_match, similarity = self.model.compare_with_reference(best_face, ref_img, threshold=match_threshold)
                        
                        if is_match:
                            is_authorized = True
                            matched_user = username
                            logger.info(f"Authorized entry: {username} detected at camera {camera_name} (similarity: {similarity:.2f})")
                            
                            # Track this person as authorized
                            self._get_or_create_tracker(frame, bbox_xywh, unauthorized=False)

                            # Send notification to frontend
                            if self.websocket_manager:
                                notification_data = {
                                    "type": "authorized_entry",
                                    "camera_id": camera_id,
                                    "camera_name": camera_name,
                                    "user_name": matched_user,
                                    "timestamp": datetime.now().isoformat(),
                                    "message": f"Authorized entry: {matched_user} at {camera_name} (Method: Standard)"
                                }
                                try:
                                    self.websocket_manager.broadcast(notification_data)
                                    logger.info(f"Sent authorized_entry notification for {matched_user} via WebSocket.")
                                except Exception as e:
                                    logger.error(f"Failed to send authorized_entry notification via WebSocket: {e}")
                            break
            
            # If person not authorized, handle as unauthorized entry
            if not is_authorized:
                # Get or create tracker for this person
                person_id = self._get_or_create_tracker(frame, bbox_xywh, unauthorized=True)
                current_time = time.time()
                person_tracker = self.person_trackers[person_id]
                
                # Check if we already have an image for this person today
                person_has_face_today = False
                for log in self.unauthorized_logs:
                    if log.get("person_id") == person_id and log.get("date") == today and log.get("has_face", False):
                        person_has_face_today = True
                        break
                
                # Save face image if it's good quality (or even moderate quality now)
                # Lowering face quality threshold to accept more faces
                if best_face is not None and face_quality >= 0.2:  # Lowered from 0.3+ to 0.2+
                    if not person_has_face_today or face_quality > person_tracker.get("best_face_quality", 0) + 0.1:  # Lowered quality improvement threshold
                        detection_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Save enhanced face image with a unique identifier
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
                time_since_first_detection = current_time - person_tracker.get("first_seen", 0)
                last_logged = person_tracker.get("last_logged", 0)
                time_since_last_log = current_time - last_logged
                
                # Check if the person has left and returned
                person_returned = person_id in self.person_presence_state and self.person_presence_state[person_id]
                
                # Use more lenient conditions to create the unauthorized entry notification
                should_log = (time_since_first_detection > 3.0 and  # Reduced from 5.0 to 3.0 seconds
                            (last_logged == 0 or  # Never logged before
                              (time_since_last_log > self.detection_interval and  # Sufficient time passed
                              (not person_has_face_today or person_returned))))  # Don't have face for today or person returned
                
                if should_log:
                    # Always log unauthorized entries
                    best_face_path = None
                    best_face_quality = 0.0
                    
                    # Check for best face image
                    if person_id in self.daily_persons.get(today, {}):
                        best_face_path = self.daily_persons[today][person_id].get("face_path")
                        best_face_quality = self.daily_persons[today][person_id].get("face_quality", 0)
                    else:
                        best_face_path = person_tracker.get("best_face_path")
                        best_face_quality = person_tracker.get("best_face_quality", 0)
                    
                    # Create unauthorized log entry
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
                    
                    # Save context frame for debugging
                    if self.debug_mode:
                        detection_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        context_filename = f"context_{person_id}_{today}_{detection_time}.jpg"
                        context_path = os.path.join(self.unauthorized_images_dir, context_filename)
                        frame_with_bbox = frame.copy()
                        cv2.rectangle(frame_with_bbox, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        self._save_frame(frame_with_bbox, context_path)
                        logger.info(f"Saved unauthorized entry context image to {context_path}")
                    
                    # Send notification to frontend
                    if self.websocket_manager:
                        notification_data = {
                            "type": "unauthorized_entry",
                            "camera_id": unauthorized_log["camera_id"],
                            "camera_name": unauthorized_log["camera_name"],
                            "person_id": unauthorized_log["person_id"],
                            "timestamp": unauthorized_log["timestamp"],
                            "face_path": unauthorized_log["face_path"], 
                            "face_quality": unauthorized_log["face_quality"],
                            "has_face": unauthorized_log["has_face"],
                            "is_front_facing": unauthorized_log.get("is_front_facing", False),
                            "detection_method": unauthorized_log.get("detection_method", "standard"),
                            "message": f"Unauthorized entry detected at {unauthorized_log['camera_name']}"
                        }
                        try:
                            self.websocket_manager.broadcast(notification_data)
                            logger.info(f"Sent unauthorized_entry notification for person {unauthorized_log['person_id']} via WebSocket.")
                        except Exception as e:
                            logger.error(f"Failed to send unauthorized_entry notification via WebSocket: {e}")
        
        # Return whether any unauthorized persons were detected in this frame
        return unauthorized_detected