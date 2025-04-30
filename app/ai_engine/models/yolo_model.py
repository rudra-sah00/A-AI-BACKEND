import cv2
import logging
import os
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional

# Import torch and YOLO related modules
try:
    import torch
    from ultralytics import YOLO
except ImportError:
    logging.error("Failed to import torch or YOLO. Please ensure they are installed.")

# Import face recognition libraries if available
try:
    import dlib
    import face_recognition
    HAVE_FACE_RECOGNITION = True
except ImportError:
    HAVE_FACE_RECOGNITION = False
    logging.warning("Face recognition libraries not available, will use basic comparison methods")

logger = logging.getLogger(__name__)

class YOLOModel:
    """YOLO model for object detection and person recognition"""
    
    def __init__(self, model_name="yolov8n.pt"):
        self.model_name = model_name
        self.model = None
        self.device = "cpu"  # Default to CPU
        self.last_inference_time = 0
        self.load_model()
        
        # Initialize face recognition model if available
        if HAVE_FACE_RECOGNITION:
            logger.info("Using advanced face recognition capabilities")
        else:
            logger.info("Using basic face comparison methods")
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            # Check for GPU
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA for YOLO model")
            else:
                logger.info("Using CPU for YOLO model")
            
            # Load YOLO model
            self.model = YOLO(self.model_name)
            logger.info(f"Loaded YOLO model: {self.model_name}")
            
            # Run a warmup inference
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_image)
            logger.info("YOLO model warmup complete")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            return False
    
    def detect_persons(self, frame) -> List[Dict]:
        """
        Detect persons in the frame
        Returns a list of detections with bounding boxes
        """
        if self.model is None:
            if not self.load_model():
                return []
        
        try:
            # Record inference time
            start_time = time.time()
            
            # Use full resolution image for detection (no resizing)
            # If memory becomes an issue, we can introduce a configurable parameter
            results = self.model(frame, classes=0, verbose=False)  # Class 0 is person in COCO
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get confidence
                    confidence = float(box.conf[0])
                    
                    # Get class
                    class_id = int(box.cls[0])
                    
                    # Add detection
                    detection = {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": confidence,
                        "class_id": class_id
                    }
                    detections.append(detection)
            
            # Update inference time
            self.last_inference_time = time.time() - start_time
            
            # Log detection information for debugging
            if detections:
                logger.debug(f"Detected {len(detections)} person(s) in {self.last_inference_time:.3f}s")
            
            return detections
        
        except Exception as e:
            logger.error(f"Error in person detection: {str(e)}")
            return []
    
    def extract_person_image(self, frame, bbox) -> Optional[np.ndarray]:
        """Extract person image from frame using bounding box"""
        try:
            x1, y1, x2, y2 = bbox
            person_img = frame[y1:y2, x1:x2]
            return person_img
        except Exception as e:
            logger.error(f"Error extracting person image: {str(e)}")
            return None

    def compare_with_reference(self, person_img, reference_img, threshold=0.6) -> Tuple[bool, float]:
        """
        Compare detected person with reference image to identify the person
        Returns a tuple of (is_match, similarity_score)
        
        Uses advanced face recognition if available, otherwise falls back to basic comparison
        """
        try:
            # Use face_recognition library if available for better accuracy
            if HAVE_FACE_RECOGNITION:
                return self._advanced_face_compare(person_img, reference_img, threshold)
            else:
                return self._basic_image_compare(person_img, reference_img, threshold)
                
        except Exception as e:
            logger.error(f"Error comparing images: {str(e)}")
            return False, 0.0
            
    def _advanced_face_compare(self, face_img, reference_img, threshold=0.6) -> Tuple[bool, float]:
        """
        Advanced face comparison using face_recognition library
        Creates facial embeddings and computes distance for accurate matching
        """
        try:
            # Convert BGR images to RGB (face_recognition expects RGB)
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            ref_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
            
            # Get face encodings (embeddings)
            # The model parameter can be:
            # - 'large': more accurate but slower
            # - 'small': faster but less accurate
            # We'll use 'small' for real-time performance, but can be changed to 'large'
            face_encoding = face_recognition.face_encodings(face_rgb, model="small")
            ref_encoding = face_recognition.face_encodings(ref_rgb, model="small")
            
            # Check if encodings were found
            if not face_encoding or not ref_encoding:
                logger.debug("No face encoding found in one or both images")
                # Fall back to basic comparison
                return self._basic_image_compare(face_img, reference_img, threshold)
            
            # Compute face distance
            face_distance = face_recognition.face_distance(ref_encoding, face_encoding[0])
            
            # Convert distance to similarity score (1 - distance)
            similarity = 1 - float(face_distance[0])
            
            # Adjust threshold for face recognition distances (typically 0.6 is a good value)
            adjusted_threshold = 1 - threshold
            is_match = face_distance[0] <= adjusted_threshold
            
            logger.debug(f"Face comparison: distance={face_distance[0]:.4f}, similarity={similarity:.4f}, match={is_match}")
            
            return is_match, similarity
            
        except Exception as e:
            logger.warning(f"Advanced face comparison failed: {str(e)}, falling back to basic comparison")
            return self._basic_image_compare(face_img, reference_img, threshold)
    
    def _basic_image_compare(self, person_img, reference_img, threshold=0.6) -> Tuple[bool, float]:
        """
        Basic image comparison using multiple methods for better accuracy
        - Histogram comparison
        - ORB feature matching
        - Structural similarity
        """
        try:
            # Resize both images to same dimensions for comparison
            height = 256
            width = int(height * person_img.shape[1] / person_img.shape[0])
            person_img = cv2.resize(person_img, (width, height))
            reference_img = cv2.resize(reference_img, (width, height))
            
            # 1. HISTOGRAM COMPARISON
            # Convert to grayscale
            person_gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
            
            # Calculate histograms
            person_hist = cv2.calcHist([person_gray], [0], None, [256], [0, 256])
            reference_hist = cv2.calcHist([reference_gray], [0], None, [256], [0, 256])
            
            # Normalize histograms
            cv2.normalize(person_hist, person_hist, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(reference_hist, reference_hist, 0, 1, cv2.NORM_MINMAX)
            
            # Compare histograms (correlation method)
            hist_similarity = cv2.compareHist(person_hist, reference_hist, cv2.HISTCMP_CORREL)
            
            # 2. ORB FEATURE MATCHING
            # Create ORB detector
            orb = cv2.ORB_create()
            
            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(person_gray, None)
            kp2, des2 = orb.detectAndCompute(reference_gray, None)
            
            # Feature matching
            orb_similarity = 0.0
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                # Create BFMatcher (Brute Force Matcher)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                
                # Match descriptors
                matches = bf.match(des1, des2)
                
                # Sort by distance
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Calculate similarity based on number of good matches
                good_matches = [m for m in matches if m.distance < 50]  # Lower distance is better
                orb_similarity = len(good_matches) / max(len(kp1), len(kp2), 1) if len(kp1) > 0 and len(kp2) > 0 else 0
            
            # 3. COLOR HISTOGRAM COMPARISON (additional to grayscale)
            # Calculate color histograms
            hsv_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
            hsv_reference = cv2.cvtColor(reference_img, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for H and S channels
            h_bins = 30
            s_bins = 32
            histSize = [h_bins, s_bins]
            h_ranges = [0, 180]
            s_ranges = [0, 256]
            ranges = h_ranges + s_ranges
            channels = [0, 1]  # Use H and S channels
            
            person_hist_color = cv2.calcHist([hsv_person], channels, None, histSize, ranges, accumulate=False)
            reference_hist_color = cv2.calcHist([hsv_reference], channels, None, histSize, ranges, accumulate=False)
            
            cv2.normalize(person_hist_color, person_hist_color, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(reference_hist_color, reference_hist_color, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Compare color histograms
            color_similarity = cv2.compareHist(person_hist_color, reference_hist_color, cv2.HISTCMP_CORREL)
            
            # Calculate weighted average of all metrics
            # Give more weight to feature matching which is more discriminative
            final_similarity = 0.3 * hist_similarity + 0.4 * orb_similarity + 0.3 * color_similarity
            is_match = final_similarity >= threshold
            
            logger.debug(f"Basic image comparison: hist={hist_similarity:.4f}, orb={orb_similarity:.4f}, " +
                       f"color={color_similarity:.4f}, final={final_similarity:.4f}")
            
            return is_match, final_similarity
        
        except Exception as e:
            logger.error(f"Basic image comparison error: {str(e)}")
            return False, 0.0
            
    def enhance_face_image(self, face_img) -> np.ndarray:
        """
        Enhance the face image for better recognition
        """
        if face_img is None or face_img.size == 0:
            return face_img
            
        try:
            # Convert to float for better precision in operations
            face_float = face_img.astype(np.float32) / 255.0
            
            # Apply adjustments
            # 1. Contrast enhancement
            alpha = 1.3  # Contrast control (1.0 means no change)
            beta = 0.0   # Brightness control (0 means no change)
            
            # Apply contrast
            face_adjusted = cv2.convertScaleAbs(face_float, alpha=alpha, beta=beta)
            
            # 2. Sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
            face_sharp = cv2.filter2D(face_adjusted, -1, kernel)
            
            # 3. Noise reduction
            face_denoised = cv2.fastNlMeansDenoisingColored(
                (face_sharp * 255).astype(np.uint8), None, 5, 5, 7, 21)
            
            return face_denoised
            
        except Exception as e:
            logger.warning(f"Face enhancement failed: {str(e)}")
            return face_img