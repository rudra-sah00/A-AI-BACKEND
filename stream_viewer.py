#!/usr/bin/env python3
"""
Stream Viewer - Real-time RTSP stream viewer with person detection visualization
Uses the existing AI engine modules for detection and authorization logic
"""
import os
import sys
import cv2
import json
import time
import argparse
import numpy as np
from datetime import datetime
import logging
from threading import Thread

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("stream_viewer")

# Import modules from the main app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.ai_engine.utils.face_detector import FaceDetector
from app.ai_engine.models.yolo_model import YOLOModel
from app.utils.stream_validator import StreamValidator

# Try to import InsightFace for enhanced detection
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAVE_INSIGHTFACE = True
except ImportError:
    HAVE_INSIGHTFACE = False
    logger.warning("InsightFace library not available, face detection quality may be reduced")

class StreamViewer:
    """
    Real-time RTSP stream viewer with AI detection and authorization visualization
    """
    def __init__(self, camera_id=None, rtsp_url=None, authorized_role="admin", display_width=1280, display_height=720):
        """
        Initialize the stream viewer
        
        Args:
            camera_id: ID of the camera to use from cameras.json file
            rtsp_url: Direct RTSP URL to use (alternative to camera_id)
            authorized_role: Role to consider as authorized (default: admin)
            display_width: Width of the display window
            display_height: Height of the display window
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.authorized_role = authorized_role
        self.display_width = display_width
        self.display_height = display_height
        
        # Window name
        self.window_name = "AI Stream Viewer"
        
        # Load configuration
        self._load_config()
        
        # Initialize models
        self.yolo_model = YOLOModel()
        self.face_detector = FaceDetector(min_face_size=(50, 50))
        
        # Initialize InsightFace if available
        self.insightface_analyzer = None
        if HAVE_INSIGHTFACE:
            try:
                self.insightface_analyzer = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider']
                )
                self.insightface_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("Initialized InsightFace for enhanced face recognition")
            except Exception as e:
                logger.warning(f"Failed to initialize InsightFace: {e}")
                self.insightface_analyzer = None
        
        # Performance tracking
        self.fps_history = []
        self.fps_avg_count = 10
        self.last_frame_time = time.time()
        self.frame_count = 0
        
        # Person tracking
        self.person_trackers = {}
        self.track_expiration_time = 5.0  # Seconds until a tracked person is forgotten
        self.authorized_cache = {}  # Cache authorization results for better performance
        self.unauthorized_person_images = set()  # Track which unauthorized persons we've already saved images for
        
        # Debug mode - set to False to disable saving images
        self.debug_mode = False  # Disable debug image saving by default
        
        # Save frequency settings
        self.save_every_n_frames = 300  # Only save debug images every 300 frames (10 seconds at 30fps)
        self.last_save_frame = 0
        
        # Reference faces for authorized users
        self.reference_images = self._load_reference_images()
        
        logger.info(f"Stream viewer initialized with {len(self.reference_images)} reference images")
    
    def _load_config(self):
        """Load configuration from JSON files"""
        # Load cameras data
        cameras_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cameras.json")
        users_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users.json")
        rules_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rules.json")
        
        self.cameras = {}
        self.users = {}
        self.rules = {}
        self.has_authorization_rules = False
        
        try:
            if os.path.exists(cameras_file):
                with open(cameras_file, 'r') as f:
                    self.cameras = json.load(f)
                logger.info(f"Loaded {len(self.cameras)} cameras from {cameras_file}")
            
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    self.users = json.load(f)
                logger.info(f"Loaded {len(self.users)} users from {users_file}")
            
            # Load rules for authorization
            if os.path.exists(rules_file):
                with open(rules_file, 'r') as f:
                    self.rules = json.load(f)
                logger.info(f"Loaded {len(self.rules)} rules from {rules_file}")
                
                # Check if any rules apply to this camera and are for authorized_entry events
                for rule_id, rule in self.rules.items():
                    if (rule.get("enabled", True) and 
                        rule.get("event") == "authorized_entry" and 
                        (rule.get("cameraId") == self.camera_id or not self.camera_id) and
                        rule.get("condition", {}).get("data", {}).get("role") == self.authorized_role):
                        self.has_authorization_rules = True
                        logger.info(f"Found active authorization rule: {rule.get('name')}")
                        break
                
                if not self.has_authorization_rules:
                    logger.info(f"No active authorization rules found for role: {self.authorized_role} and camera: {self.camera_id}")
                
            # Get RTSP URL from camera_id if provided
            if self.camera_id and not self.rtsp_url:
                if self.camera_id in self.cameras:
                    self.rtsp_url = self.cameras[self.camera_id].get("rtsp_url")
                    self.camera_name = self.cameras[self.camera_id].get("name", f"Camera {self.camera_id}")
                    logger.info(f"Using camera: {self.camera_name} with URL: {self.rtsp_url}")
                else:
                    logger.error(f"Camera ID {self.camera_id} not found in cameras.json")
                    sys.exit(1)
            
            # Use camera name from provided ID, or create a name based on the URL
            if not hasattr(self, 'camera_name'):
                self.camera_name = f"Stream {self.rtsp_url.split('/')[-1]}" if self.rtsp_url else "Unknown Camera"
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def _load_reference_images(self) -> dict:
        """Load all reference images for authorized users"""
        reference_images = {}
        
        for username, user_data in self.users.items():
            # Only load for users with the authorized role
            if self.authorized_role and user_data.get("role", "").lower() != self.authorized_role.lower():
                continue
                
            photo_path = user_data.get("photo_path")
            if not photo_path:
                continue
                
            # Convert relative path to absolute path if needed
            if not os.path.isabs(photo_path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                photo_path = os.path.join(base_dir, photo_path)
                
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
                                "quality": 0.5  # Moderate quality for full images
                            }
                            logger.warning(f"No good face detected for {username}, using full image with moderate quality")
                    else:
                        # If no face detected, use the whole image
                        reference_images[username] = {
                            "image": image,
                            "user_data": user_data,
                            "quality": 0.5  # Moderate quality for full images
                        }
                        logger.warning(f"No face detected for {username}, using full image with moderate quality")
                except Exception as e:
                    logger.error(f"Error loading reference image for {username}: {str(e)}")
        
        logger.info(f"Loaded {len(reference_images)} reference images for role: {self.authorized_role}")
        return reference_images
    
    def _compare_with_insightface(self, face_img, reference_data):
        """Compare a face with reference data using InsightFace for better accuracy"""
        if not HAVE_INSIGHTFACE or self.insightface_analyzer is None:
            return False, 0.0, None
            
        try:
            # Check if we already have embedding in the reference data
            ref_embedding = reference_data.get("embedding")
            
            if ref_embedding is None:
                # Get embedding from reference image
                ref_img = reference_data["image"]
                ref_faces = self.insightface_analyzer.get(ref_img)
                
                if len(ref_faces) == 0:
                    return False, 0.0, None
                    
                ref_embedding = ref_faces[0].embedding
            
            # Get embedding for detected face
            faces = self.insightface_analyzer.get(face_img)
            
            if len(faces) == 0:
                return False, 0.0, None
                
            test_embedding = faces[0].embedding
            
            # Calculate cosine similarity
            similarity = np.dot(ref_embedding, test_embedding) / (np.linalg.norm(ref_embedding) * np.linalg.norm(test_embedding))
            
            # Consider it a match if similarity exceeds threshold
            # Threshold is lower here for more permissive matching in the viewer
            threshold = 0.4
            is_match = similarity > threshold
            
            # Return match status, similarity score, and matched embedding
            return is_match, similarity, ref_embedding
            
        except Exception as e:
            logger.warning(f"InsightFace comparison error: {e}")
            return False, 0.0, None
    
    def _is_authorized_person(self, person_img, person_id=None):
        """
        Check if a person is authorized by matching their face against reference images
        Returns a tuple of (is_authorized, matched_username, confidence)
        """
        # Check cache first for better performance
        if person_id in self.authorized_cache:
            cache_entry = self.authorized_cache[person_id]
            # Use cached result if it's recent (within 3 seconds)
            if time.time() - cache_entry["timestamp"] < 3.0:
                return cache_entry["is_authorized"], cache_entry["username"], cache_entry["confidence"]
        
        # Try to detect face
        best_face = None
        best_rect = None
        face_quality = 0.0
        is_insightface_detection = False
        
        # Try InsightFace first
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
            except Exception as e:
                logger.debug(f"InsightFace detection error: {e}")
        
        # Fall back to regular face detection - use more permissive parameters for glasses
        if best_face is None:
            try:
                # Lower minimum validation score to better handle glasses and lighting variations
                faces = self.face_detector.detect_faces(person_img, min_validation_score=0.1)
                if faces:
                    best_face, best_rect, face_quality = self.face_detector.get_best_face(person_img, faces)
            except Exception as e:
                logger.debug(f"Face detection error: {e}")
        
        # If still no face found, use the whole upper portion of the image
        # This helps in cases where the face detector fails due to glasses or lighting
        if best_face is None:
            h, w = person_img.shape[:2]
            # Use top 40% of the person image as an approximate face region
            face_h = int(h * 0.4)
            best_face = person_img[0:face_h, 0:w].copy()
            best_rect = (0, 0, w, face_h)
            face_quality = 0.3  # Moderate quality since we're approximating
            logger.debug("Using upper portion of person image as face approximation")
        
        # Compare with reference images
        best_match = None
        best_confidence = 0.0
        
        for username, ref_data in self.reference_images.items():
            # Try InsightFace comparison first
            if is_insightface_detection and HAVE_INSIGHTFACE and self.insightface_analyzer:
                is_match, similarity, _ = self._compare_with_insightface(best_face, ref_data)
                
                if is_match and similarity > best_confidence:
                    best_match = username
                    best_confidence = similarity
            else:
                # More permissive threshold for the viewer - lowered from 0.4 to 0.3
                match_threshold = 0.3
                ref_img = ref_data["image"]
                
                try:
                    is_match, similarity = self.yolo_model.compare_with_reference(best_face, ref_img, threshold=match_threshold)
                    
                    if is_match and similarity > best_confidence:
                        best_match = username
                        best_confidence = similarity
                except Exception as e:
                    logger.debug(f"Face comparison error: {e}")
        
        is_authorized = best_match is not None
        
        # Cache the result
        if person_id is not None:
            self.authorized_cache[person_id] = {
                "is_authorized": is_authorized,
                "username": best_match,
                "confidence": best_confidence,
                "timestamp": time.time()
            }
        
        return is_authorized, best_match, best_confidence
    
    def _cleanup_expired_trackers(self):
        """Remove expired person trackers"""
        current_time = time.time()
        expired_ids = []
        
        for person_id, tracker_data in self.person_trackers.items():
            last_seen = tracker_data.get("last_seen", 0)
            if current_time - last_seen > self.track_expiration_time:
                expired_ids.append(person_id)
        
        # Remove expired trackers
        for person_id in expired_ids:
            del self.person_trackers[person_id]
            # Also remove from auth cache
            if person_id in self.authorized_cache:
                del self.authorized_cache[person_id]
    
    def _calculate_fps(self):
        """Calculate and return the current FPS"""
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        
        # Avoid division by zero
        if time_diff <= 0:
            return 0
            
        fps = 1.0 / time_diff
        self.fps_history.append(fps)
        
        # Keep only the most recent frames for averaging
        if len(self.fps_history) > self.fps_avg_count:
            self.fps_history.pop(0)
            
        # Calculate average FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Update last frame time
        self.last_frame_time = current_time
        self.frame_count += 1
        
        return avg_fps
    
    def _is_same_person(self, bbox1, bbox2, frame_width, frame_height, threshold=0.5):
        """
        Determine if two bounding boxes likely belong to the same person
        Uses intersection over union (IoU) and relative positions
        """
        # Calculate IoU
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        
        # Convert (x,y,w,h) to (x1,y1,x2,y2)
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
    
    def _get_or_create_tracker(self, frame, bbox):
        """
        Get an existing tracker for a person or create a new one
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
                return person_id
        
        # If no match found, create a new tracker
        import uuid
        person_id = str(uuid.uuid4())
        self.person_trackers[person_id] = {
            "id": person_id,
            "bbox": bbox,
            "first_seen": current_time,
            "last_seen": current_time
        }
        
        return person_id
    
    def process_frame(self, frame):
        """
        Process a frame from the stream
        
        Args:
            frame: The frame to process
        
        Returns:
            Processed frame with visualizations
        """
        if frame is None or frame.size == 0:
            return frame
            
        # Make a copy for drawing
        result_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate FPS
        fps = self._calculate_fps()
        
        # Clean up expired trackers
        self._cleanup_expired_trackers()
        
        # Detect persons in the frame
        detections = self.yolo_model.detect_persons(frame)
        
        if detections:
            for detection in detections:
                bbox = detection["bbox"]
                confidence = detection["confidence"]
                
                # Convert YOLO format (x1, y1, x2, y2) to (x, y, w, h)
                x1, y1, x2, y2 = bbox
                person_width = x2 - x1
                person_height = y2 - y1
                bbox_xywh = (int(x1), int(y1), int(person_width), int(person_height))
                
                # Skip low confidence detections
                if confidence < 0.4:
                    continue
                
                # Get or create tracker for this person
                person_id = self._get_or_create_tracker(frame, bbox_xywh)
                
                # Extract person image
                person_img = frame[int(y1):int(y2), int(x1):int(x2)].copy()
                if person_img.size == 0 or person_img.shape[0] == 0 or person_img.shape[1] == 0:
                    continue
                
                # Check if person is authorized
                is_authorized, matched_user, confidence = self._is_authorized_person(person_img, person_id)
                
                # Determine display style based on authorization and rules
                text = ""
                color = (255, 255, 0)  # Default yellow for detected people with no rules
                
                if is_authorized:
                    color = (0, 255, 0)  # Green for authorized
                    text = f"{matched_user} ({confidence:.2f})"
                elif self.has_authorization_rules:
                    # Only mark as unauthorized if we have explicit rules requiring authorization
                    color = (0, 0, 255)  # Red for unauthorized
                    text = "Unauthorized"
                else:
                    # If no rules, just show person detection without unauthorized label
                    color = (255, 255, 0)  # Yellow for person detection without rules
                    text = "Person"
                
                # Draw rectangle and text
                cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Add a black background for text for better readability
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(result_frame, 
                            (int(x1), int(y1) - text_size[1] - 10), 
                            (int(x1) + text_size[0], int(y1)), 
                            (0, 0, 0), 
                            -1)
                cv2.putText(result_frame, 
                          text, 
                          (int(x1), int(y1) - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, 
                          (255, 255, 255), 
                          2)
        
        # Draw camera info and FPS
        cv2.putText(result_frame, 
                  f"{self.camera_name} - FPS: {fps:.2f}", 
                  (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 
                  0.7, 
                  (255, 255, 255), 
                  2)
        
        # Draw rules status
        rule_status = "Rules Active" if self.has_authorization_rules else "No Authorization Rules"
        rule_color = (0, 255, 0) if self.has_authorization_rules else (0, 165, 255)  # Green or orange
        
        cv2.putText(result_frame, 
                  rule_status, 
                  (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 
                  0.7, 
                  rule_color, 
                  2)
        
        return result_frame
    
    def start(self):
        """Start the stream viewer"""
        # Validate the RTSP stream
        if not self.rtsp_url:
            logger.error("No RTSP URL provided")
            return False
            
        logger.info(f"Opening stream: {self.rtsp_url}")
        
        # Validate stream accessibility
        stream_valid = StreamValidator.validate_rtsp_stream(self.rtsp_url)
        if not stream_valid.get("is_valid", False):
            logger.error(f"Could not access stream: {self.rtsp_url}")
            logger.error(f"Error: {stream_valid.get('error_message', 'Unknown error')}")
            return False
        
        # Create a window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        
        # Set FFMPEG options for better stream handling
        # These options improve handling of corrupted frames and network issues
        rtsp_url_with_options = self.rtsp_url
        
        # Add FFMPEG options directly to the URL as a query string for transport method
        if "?" not in rtsp_url_with_options:
            rtsp_url_with_options += "?rtsp_transport=tcp"
        else:
            rtsp_url_with_options += "&rtsp_transport=tcp"
        
        # Open the stream with only two arguments (compatible with OpenCV 4.11.0)
        cap = cv2.VideoCapture(rtsp_url_with_options, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            logger.error(f"Could not open stream: {rtsp_url_with_options}")
            return False
        
        # Additional stream configuration
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Reduce buffer size for lower latency
        
        # Frame processing variables
        consecutive_errors = 0
        max_consecutive_errors = 10
        last_successful_frame = None
        
        # Start processing frames
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_errors += 1
                    logger.warning(f"Failed to read frame from stream (error {consecutive_errors}/{max_consecutive_errors})")
                    
                    # Use the last good frame if available to prevent display interruption
                    if last_successful_frame is not None:
                        processed_frame = self.process_frame(last_successful_frame.copy())
                        cv2.putText(processed_frame, 
                                  "Stream Error - Reconnecting...", 
                                  (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, 
                                  (0, 0, 255), 
                                  2)
                        cv2.imshow(self.window_name, processed_frame)
                    
                    # Check for key press to exit even during reconnection attempts
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        break
                    
                    # Retry up to max_consecutive_errors before attempting a full reconnection
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Too many consecutive errors ({consecutive_errors}), reconnecting...")
                        cap.release()
                        time.sleep(2)  # Wait before reconnecting
                        
                        # Reconstruct VideoCapture with updated URL
                        cap = cv2.VideoCapture(rtsp_url_with_options, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                        consecutive_errors = 0
                    
                    # Wait a bit before retrying
                    time.sleep(0.1)
                    continue
                
                # Successfully read a frame, reset error counter
                consecutive_errors = 0
                last_successful_frame = frame.copy()
                
                # Resize for better performance if needed
                if frame.shape[1] > 1280:
                    scale_factor = 1280 / frame.shape[1]
                    frame = cv2.resize(frame, (1280, int(frame.shape[0] * scale_factor)))
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Show the processed frame
                cv2.imshow(self.window_name, processed_frame)
                
                # Check for key press to exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting")
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Stream viewer stopped")
        
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AI Stream Viewer for RTSP cameras")
    parser.add_argument("--camera-id", help="Camera ID from cameras.json")
    parser.add_argument("--rtsp-url", help="Direct RTSP URL to use")
    parser.add_argument("--role", default="admin", help="Role to consider as authorized (default: admin)")
    parser.add_argument("--width", type=int, default=1280, help="Display window width")
    parser.add_argument("--height", type=int, default=720, help="Display window height")
    
    args = parser.parse_args()
    
    if not args.camera_id and not args.rtsp_url:
        parser.error("Either --camera-id or --rtsp-url must be provided")
    
    # Create and start the viewer
    viewer = StreamViewer(
        camera_id=args.camera_id,
        rtsp_url=args.rtsp_url,
        authorized_role=args.role,
        display_width=args.width,
        display_height=args.height
    )
    
    viewer.start()

if __name__ == "__main__":
    main()