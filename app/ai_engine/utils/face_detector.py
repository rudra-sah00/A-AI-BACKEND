import cv2
import logging
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import time

# Import advanced face detection libraries
try:
    import dlib
    import face_recognition
    HAVE_FACE_RECOGNITION = True
except ImportError:
    HAVE_FACE_RECOGNITION = False
    logging.warning("face_recognition library not available, falling back to OpenCV")

# Import InsightFace for enhanced low resolution face detection
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAVE_INSIGHTFACE = True
except ImportError:
    HAVE_INSIGHTFACE = False
    logging.warning("InsightFace library not available, using alternative methods")

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Advanced face detector utility class for detecting, cropping and enhancing faces in images
    with multiple detection methods and state-of-the-art enhancement
    """
    def __init__(self, min_face_size=(30, 30), cache_size=100):
        self.min_face_size = min_face_size
        
        # Initialize face detection cache for better performance
        self.face_cache = {}  # Cache recent face encodings for fast reidentification
        self.face_cache_keys = []  # Ordered list of keys for LRU cache implementation
        self.max_cache_size = cache_size  # Maximum number of faces to cache
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize memory optimization
        self.last_cache_cleanup = time.time()
        self.cleanup_interval = 300  # Cleanup cache every 5 minutes
        self.identical_face_threshold = 0.6  # Threshold for face similarity (lower is stricter)
        
        # Initialize multiple detection methods for redundancy and improved accuracy
        self.detection_methods = []
        
        # 0. Try to use InsightFace (best for low-resolution and challenging images)
        try:
            if HAVE_INSIGHTFACE:
                # Initialize InsightFace with appropriate models
                self.insightface_app = FaceAnalysis(
                    name='buffalo_l',  # Using light model for better speed/accuracy balance
                    providers=['CPUExecutionProvider']  # Using CPU for compatibility
                )
                self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
                self.detection_methods.insert(0, self._detect_faces_insightface)
                logger.info("Initialized InsightFace detector (best for low-resolution)")
        except Exception as e:
            logger.warning(f"Failed to initialize InsightFace: {e}")
            
        # 1. Try to use dlib's HOG face detector (most accurate for profile faces)
        try:
            if HAVE_FACE_RECOGNITION:
                self.face_detector_hog = dlib.get_frontal_face_detector()
                self.detection_methods.append(self._detect_faces_dlib_hog)
                logger.info("Initialized dlib HOG face detector")
        except Exception as e:
            logger.warning(f"Failed to initialize dlib HOG face detector: {e}")
            
        # 2. Try to use dlib's CNN face detector if available (more accurate but slower)
        try:
            cnn_model_path = os.path.join(os.path.dirname(__file__), "../models/mmod_human_face_detector.dat")
            if HAVE_FACE_RECOGNITION and os.path.exists(cnn_model_path):
                self.face_detector_cnn = dlib.cnn_face_detection_model_v1(cnn_model_path)
                self.detection_methods.append(self._detect_faces_dlib_cnn)
                logger.info("Initialized dlib CNN face detector")
        except Exception as e:
            logger.warning(f"Failed to initialize dlib CNN face detector: {e}")

        # 3. Initialize OpenCV DNN face detector (good balance between speed and accuracy)
        try:
            model_file = os.path.join(os.path.dirname(__file__), "../models/deploy.prototxt")
            weights_file = os.path.join(os.path.dirname(__file__), 
                                      "../models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
            
            if os.path.exists(model_file) and os.path.exists(weights_file):
                self.face_net = cv2.dnn.readNetFromCaffe(model_file, weights_file)
                self.detection_methods.append(self._detect_faces_opencv_dnn)
                logger.info("Initialized OpenCV DNN face detector")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenCV DNN face detector: {e}")
            
        # 4. Fall back to Haar cascade as last resort
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if not self.face_cascade.empty():
                self.detection_methods.append(self._detect_faces_haar)
                logger.info("Initialized Haar cascade face detector")
        except Exception as e:
            logger.warning(f"Failed to initialize Haar cascade face detector: {e}")
            
        # 5. Special profile face detection
        try:
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            if not self.profile_cascade.empty():
                self.detection_methods.append(self._detect_faces_profile)
                logger.info("Initialized profile face detector")
        except Exception as e:
            logger.warning(f"Failed to initialize profile face detector: {e}")
            
        # 6. Add face detector optimized for blurry images
        try:
            self.alt_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
            if not self.alt_face_cascade.empty():
                self.detection_methods.append(self._detect_faces_alt)
                logger.info("Initialized alternative face detector for blurry images")
        except Exception as e:
            logger.warning(f"Failed to initialize alternative face detector: {e}")
            
        if not self.detection_methods:
            logger.error("No face detection methods available!")
            raise RuntimeError("Failed to initialize any face detection method")
            
        # 7. Initialize facial landmarks detector if available
        self.landmarks_predictor = None
        try:
            if HAVE_FACE_RECOGNITION:
                # Use dlib's shape predictor for facial landmarks
                predictor_path = os.path.join(os.path.dirname(__file__), 
                                            "../models/shape_predictor_68_face_landmarks.dat")
                if os.path.exists(predictor_path):
                    self.landmarks_predictor = dlib.shape_predictor(predictor_path)
                    logger.info("Initialized facial landmarks detector")
        except Exception as e:
            logger.warning(f"Failed to initialize facial landmarks detector: {e}")
        
        # 8. Initialize eye detector for better validation
        try:
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            logger.info("Initialized eye detector for validation")
        except Exception as e:
            logger.warning(f"Failed to initialize eye detector: {e}")
        
        # 9. Initialize nose detector for improved face validation
        try:
            self.nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
            logger.info("Initialized nose detector for validation")
        except Exception as e:
            logger.warning(f"Failed to initialize nose detector: {e}")
            
        logger.info(f"Face detector initialized with {len(self.detection_methods)} detection methods")
    
    def _detect_faces_insightface(self, image) -> List[Tuple[int, int, int, int]]:
        """Detect faces using InsightFace - excellent for low-resolution/challenging images"""
        if not HAVE_INSIGHTFACE:
            return []
        
        try:
            # InsightFace expects BGR (OpenCV default)
            faces = self.insightface_app.get(image)
            
            # Convert to (x, y, w, h) format
            return [(int(face.bbox[0]), int(face.bbox[1]), 
                    int(face.bbox[2] - face.bbox[0]), int(face.bbox[3] - face.bbox[1])) 
                    for face in faces]
        except Exception as e:
            logger.error(f"Error in InsightFace detection: {str(e)}")
            return []
    
    def _detect_faces_dlib_hog(self, image) -> List[Tuple[int, int, int, int]]:
        """Detect faces using dlib's HOG face detector"""
        if not HAVE_FACE_RECOGNITION:
            return []
            
        try:
            # Convert to RGB (dlib expects RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
                
            # Detect faces
            faces = self.face_detector_hog(rgb_image, 1)  # 1 = upsample once for better detection
            
            # Convert to (x, y, w, h) format
            return [(face.left(), face.top(), face.width(), face.height()) for face in faces]
        except Exception as e:
            logger.error(f"Error in dlib HOG face detection: {str(e)}")
            return []
            
    def _detect_faces_dlib_cnn(self, image) -> List[Tuple[int, int, int, int]]:
        """Detect faces using dlib's CNN face detector"""
        if not hasattr(self, 'face_detector_cnn'):
            return []
            
        try:
            # Convert to RGB (dlib expects RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
                
            # Detect faces
            faces = self.face_detector_cnn(rgb_image, 1)  # 1 = upsample once for better detection
            
            # Convert to (x, y, w, h) format
            return [(face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()) 
                    for face in faces]
        except Exception as e:
            logger.error(f"Error in dlib CNN face detection: {str(e)}")
            return []
    
    def _detect_faces_opencv_dnn(self, image) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV DNN face detector"""
        if not hasattr(self, 'face_net'):
            return []
            
        try:
            height, width = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 
                1.0,
                (300, 300), 
                [104, 117, 123], 
                False, 
                False
            )
            
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            # Process detections
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter by confidence - increased from 0.7 to 0.8 for more precision
                if confidence > 0.8:
                    # Get coordinates
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    
                    # Convert to (x, y, w, h) format
                    w = x2 - x1
                    h = y2 - y1
                    if w > 0 and h > 0:  # Ensure valid dimensions
                        faces.append((x1, y1, w, h))
                        
            return faces
        except Exception as e:
            logger.error(f"Error in OpenCV DNN face detection: {str(e)}")
            return []

    def _detect_faces_with_dnn(self, frame):
        """Detect faces using OpenCV's DNN face detector."""
        try:
            # Get frame dimensions
            (h, w) = frame.shape[:2]
            
            # Create a blob from the image
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 
                1.0, 
                (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            # Pass the blob through the network
            self.dnn_face_detector.setInput(blob)
            detections = self.dnn_face_detector.forward()
            
            # List to store detected faces
            face_rects = []
            
            # Loop over the detections
            for i in range(0, detections.shape[2]):
                # Extract the confidence of the detection
                confidence = detections[0, 0, i, 2]
                
                # Filter weak detections with higher threshold
                if confidence > 0.6:  # Increased from 0.5 to 0.6
                    # Calculate bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure box is within frame
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    # Exclude very small or invalid detections
                    if endX - startX > self.min_face_size[0] and endY - startY > self.min_face_size[1]:
                        # Store as (x, y, w, h) format
                        face_rects.append((startX, startY, endX - startX, endY - startY))
            
            return face_rects
        except Exception as e:
            logger.warning(f"DNN face detection error: {str(e)}")
            return []
    
    def _detect_faces_haar(self, image) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascade (fallback method)"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with more restrictive parameters for better precision
            # Increased minNeighbors from 3 to 5 to reduce false positives
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=5,    # Higher threshold to reduce false positives
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return list(faces)
        except Exception as e:
            logger.error(f"Error in Haar cascade face detection: {str(e)}")
            return []
            
    def _detect_faces_profile(self, image) -> List[Tuple[int, int, int, int]]:
        """Detect profile (side) faces using specialized cascade"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with increased minNeighbors for better precision
            faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,  # Increased from 3 to 4
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Also try with flipped image to detect profiles facing the other direction
            flipped = cv2.flip(gray, 1)
            flipped_faces = self.profile_cascade.detectMultiScale(
                flipped,
                scaleFactor=1.1,
                minNeighbors=4,  # Increased from 3 to 4
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Adjust coordinates for flipped faces
            width = gray.shape[1]
            for (x, y, w, h) in flipped_faces:
                # Convert coordinates back to original image
                faces = np.vstack((faces, np.array([(width - x - w, y, w, h)]))) if len(faces) else np.array([(width - x - w, y, w, h)])
            
            return list(faces)
        except Exception as e:
            logger.error(f"Error in profile face detection: {str(e)}")
            return []
    
    def _detect_faces_alt(self, image) -> List[Tuple[int, int, int, int]]:
        """Detect faces using alternative Haar cascade optimized for blurry images"""
        try:
            # First apply image enhancement for blurry images
            enhanced = self._preprocess_blurry_image(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            
            # Use more restrictive parameters for blurry images
            faces = self.alt_face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.03,
                minNeighbors=4,    # Increased from 2 to 4
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return list(faces)
        except Exception as e:
            logger.error(f"Error in alternative face detection: {str(e)}")
            return []
    
    def _preprocess_blurry_image(self, image):
        """Special preprocessing for blurry, low-quality images"""
        try:
            # Check if the image is already clear enough
            laplacian_var = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            
            # If image is already sharp enough, don't process it further
            if laplacian_var > 100:
                return image
                
            # For blurry images, apply a series of enhancements:
            
            # 1. Denoising with careful parameters to not lose details
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
            
            # 2. Apply a subtle sharpening using unsharp mask
            gaussian = cv2.GaussianBlur(denoised, (0, 0), 3.0)
            enhanced = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
            
            # 3. Enhance contrast locally to help feature detection
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Image preprocessing for blurry images failed: {str(e)}")
            return image
    
    def _is_valid_face(self, frame, rect):
        """
        Comprehensive face validation method to filter out non-face objects
        Returns a score between 0.0 and 1.0, where higher means more likely to be a face
        """
        x, y, w, h = rect
        
        # Skip if dimensions are invalid
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            return 0.0
            
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return 0.0
            
        # Skip if aspect ratio is highly unusual for a face (made stricter)
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:  # Changed from 0.65-1.35 to 0.7-1.3
            return 0.0
            
        # Skip if face is too small (increased minimum size)
        if w < self.min_face_size[0] * 1.5 or h < self.min_face_size[1] * 1.5:  # Increased multiplier from 1.2 to 1.5
            return 0.0
            
        # Start with base validation score
        validation_score = 0.0
        
        try:
            # Convert ROI to grayscale for feature detection
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # 1. Check for eyes - strong indicator of a face
            eyes = self.eye_cascade.detectMultiScale(
                gray_roi,
                scaleFactor=1.1,
                minNeighbors=6,  # Increased from 5 to 6 for more reliable detection
                minSize=(20, 20)
            )
            
            # Add points for each eye detected in the upper half of the face
            eye_count = 0
            for (ex, ey, ew, eh) in eyes:
                # Check that eyes are in the upper half of the face
                if ey < h/2:
                    validation_score += 0.25  # Each eye adds 0.25 to score
                    eye_count += 1
            
            # Require at least one eye for minimum validation
            if eye_count == 0:
                validation_score -= 0.4  # Increased penalty for no eyes detected (was 0.3)
            
            # 2. Check for nose - another face feature
            nose = self.nose_cascade.detectMultiScale(
                gray_roi,
                scaleFactor=1.1,
                minNeighbors=6,  # Increased from 5 to 6
                minSize=(20, 20)
            )
            
            nose_detected = False
            # Add points if a nose is detected in approximately the middle of the face
            for (nx, ny, nw, nh) in nose:
                # Check nose is roughly in the middle
                if nx > w/4 and nx+nw < 3*w/4 and ny > h/4 and ny+nh < 3*h/4:
                    validation_score += 0.25
                    nose_detected = True
                    
            # Penalize for no nose detection
            if not nose_detected:
                validation_score -= 0.3  # Increased penalty (was 0.2)
                    
            # 3. Use facial landmarks for validation if available
            landmarks_detected = False
            if self.landmarks_predictor and HAVE_FACE_RECOGNITION:
                try:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                    shape = self.landmarks_predictor(rgb_image, dlib_rect)
                    
                    # If we can detect a good number of landmarks, definitely a face
                    if shape.num_parts > 50:  # 68 is the maximum for the standard model
                        validation_score += 0.3
                        landmarks_detected = True
                    elif shape.num_parts > 30:
                        validation_score += 0.2
                        landmarks_detected = True
                    elif shape.num_parts > 0:
                        validation_score += 0.1
                        landmarks_detected = True
                except:
                    pass
            
            # 4. Check skin tone pattern
            hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_roi)
            
            # Check for typical skin hue range
            # Convert to 0-180 scale for OpenCV HSV
            skin_pixels = np.sum((h >= 0) & (h <= 30) | (h >= 150) & (h <= 180))
            skin_ratio = skin_pixels / (w * h)
            
            # Add points for appropriate skin tone ratio
            if skin_ratio > 0.6:  # Increased from 0.5 to 0.6
                validation_score += 0.25
            elif skin_ratio > 0.4:  # Increased from 0.3 to 0.4
                validation_score += 0.1
                
            # 5. Check symmetry - faces are generally symmetric
            # Split face in half and compare
            left_half = face_roi[:, :w//2]
            right_half = face_roi[:, w//2:]
            flipped_right = cv2.flip(right_half, 1)
            
            # Resize for comparison if needed
            if left_half.shape != flipped_right.shape:
                min_height = min(left_half.shape[0], flipped_right.shape[0])
                min_width = min(left_half.shape[1], flipped_right.shape[1])
                left_half = left_half[:min_height, :min_width]
                flipped_right = flipped_right[:min_height, :min_width]
            
            if left_half.size > 0 and flipped_right.size > 0:
                # Calculate symmetry score using structural similarity
                try:
                    from skimage.metrics import structural_similarity as ssim
                    left_gray = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
                    right_gray = cv2.cvtColor(flipped_right, cv2.COLOR_BGR2GRAY)
                    symmetry_score = ssim(left_gray, right_gray)
                    
                    # Add points based on symmetry
                    if symmetry_score > 0.85:  # Increased from 0.8 to 0.85
                        validation_score += 0.2
                    elif symmetry_score > 0.7:  # Increased from 0.65 to 0.7
                        validation_score += 0.1
                except:
                    # Fall back to simpler method if skimage not available
                    diff = cv2.absdiff(left_half, flipped_right)
                    diff_sum = np.sum(diff)
                    max_diff = left_half.size * 255 * 3  # maximum possible difference
                    symmetry_score = 1.0 - (diff_sum / max_diff)
                    
                    if symmetry_score > 0.85:  # Increased from 0.8 to 0.85
                        validation_score += 0.1
                        
            # Additional validation rule: require at least two facial features detected
            features_detected = sum([eye_count > 0, nose_detected, landmarks_detected])
            if features_detected < 2:
                validation_score -= 0.4  # Increased penalty from 0.3 to 0.4
            
            # Cap the validation score at 1.0
            validation_score = min(1.0, validation_score)
            
            # Logging for debugging
            logger.debug(f"Face validation: Aspect={aspect_ratio:.2f}, Eyes={eye_count}, Nose={nose_detected}, Features={features_detected}, Score={validation_score:.2f}")
            
            return validation_score
            
        except Exception as e:
            logger.warning(f"Face validation error: {str(e)}")
            return 0.0
    
    def detect_faces(self, frame, min_validation_score=0.6, use_cache=True):
        """
        Enhanced multi-method face detection with validation and caching
        Returns a list of (x,y,w,h) face rectangles
        """
        if frame is None or frame.size == 0:
            return []
            
        # Check cache for recent performance optimization
        if use_cache:
            cache_key = self._compute_frame_hash(frame)
            current_time = time.time()
            
            # Periodic cache cleanup
            if current_time - self.last_cache_cleanup > self.cleanup_interval:
                self._cleanup_cache()
                self.last_cache_cleanup = current_time
                
            # Check if we have this frame in cache
            if cache_key in self.face_cache:
                cached_result = self.face_cache[cache_key]
                
                # Check if cache entry is still valid (less than 1 second old)
                if current_time - cached_result['timestamp'] < 1.0:
                    self.cache_hits += 1
                    
                    # Move this key to end of LRU list (most recently used)
                    self.face_cache_keys.remove(cache_key)
                    self.face_cache_keys.append(cache_key)
                    
                    # Update timestamp to keep it fresh
                    self.face_cache[cache_key]['timestamp'] = current_time
                    
                    return cached_result['faces']
                
        self.cache_misses += 1
            
        # Initialize results
        faces = []
        height, width = frame.shape[:2]
        
        # Scale factor for better performance with large images
        scale_factor = 1.0
        if width > 1280 or height > 720:
            scale_factor = 0.5
            width_scaled = int(width * scale_factor)
            height_scaled = int(height * scale_factor)
            frame_scaled = cv2.resize(frame, (width_scaled, height_scaled))
        else:
            frame_scaled = frame
            
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2GRAY)
        
        # Apply various detection methods
        for detection_method in self.detection_methods:
            # Skip additional methods if we already found enough faces
            if len(faces) >= 3:
                break
                
            try:
                new_faces = detection_method(frame_scaled)
                faces.extend(new_faces)
                
                # If we found faces, break early to save processing time
                if len(faces) > 0:
                    break
                    
            except Exception as e:
                logger.warning(f"Face detection method error: {str(e)}")
        
        # Convert scaled coordinates back to original scale if needed
        if scale_factor != 1.0:
            for i in range(len(faces)):
                x, y, w, h = faces[i]
                faces[i] = (
                    int(x / scale_factor), 
                    int(y / scale_factor), 
                    int(w / scale_factor), 
                    int(h / scale_factor)
                )
        
        # Apply non-maximum suppression to remove overlapping detections
        if len(faces) > 1:
            faces_array = np.array(faces)
            faces = self._non_max_suppression(faces_array, overlap_thresh=0.3)
        
        # Final quality check with validation
        validated_faces = []
        for face_rect in faces:
            validation_score = self._is_valid_face(frame, face_rect)
            if validation_score >= min_validation_score:
                validated_faces.append(face_rect)
        
        # Update cache with the result
        if use_cache:
            # Add new result to cache
            if cache_key not in self.face_cache:
                self.face_cache[cache_key] = {
                    'faces': validated_faces,
                    'timestamp': time.time()
                }
                self.face_cache_keys.append(cache_key)
                
                # Enforce cache size limit
                if len(self.face_cache_keys) > self.max_cache_size:
                    oldest_key = self.face_cache_keys.pop(0)  # Remove oldest key
                    if oldest_key in self.face_cache:
                        del self.face_cache[oldest_key]
            else:
                # Update existing cache entry
                self.face_cache[cache_key]['faces'] = validated_faces
                self.face_cache[cache_key]['timestamp'] = time.time()
                
                # Move this key to end of LRU list
                self.face_cache_keys.remove(cache_key)
                self.face_cache_keys.append(cache_key)
        
        return validated_faces
    
    def _compute_frame_hash(self, frame):
        """Compute a fast and efficient hash for frame caching"""
        if frame.size == 0:
            return None
            
        # For performance, only use selected pixels and downsample
        h, w = frame.shape[:2]
        sample = frame[0:h:20, 0:w:20]  # Take every 20th pixel
        
        # For even faster hashing, convert to grayscale and use only center region
        if sample.ndim > 2:
            sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        else:
            sample_gray = sample
            
        # Use MD5 for fast hashing - not for security but for uniqueness
        import hashlib
        return hashlib.md5(sample_gray.tobytes()).hexdigest()
    
    def _cleanup_cache(self):
        """Remove old entries from the face detection cache"""
        current_time = time.time()
        keys_to_remove = []
        
        # Find expired cache entries (older than 5 seconds)
        for key in self.face_cache_keys:
            if current_time - self.face_cache[key]['timestamp'] > 5.0:
                keys_to_remove.append(key)
        
        # Remove expired entries
        for key in keys_to_remove:
            if key in self.face_cache:
                del self.face_cache[key]
            if key in self.face_cache_keys:
                self.face_cache_keys.remove(key)
                
        logger.debug(f"Face cache cleanup: removed {len(keys_to_remove)} entries, " +
                   f"cache now has {len(self.face_cache)} items " +
                   f"(hits: {self.cache_hits}, misses: {self.cache_misses})")
        
    def compare_face_similarity(self, face1, face2):
        """
        Compare two face images and determine if they are the same person
        
        Args:
            face1: First face image
            face2: Second face image
            
        Returns:
            Tuple of (is_same_person, similarity_score)
        """
        if face1 is None or face2 is None or face1.size == 0 or face2.size == 0:
            return False, 0.0
            
        # Try InsightFace first (most accurate for low resolution)
        if HAVE_INSIGHTFACE:
            try:
                # Extract embeddings using InsightFace
                faces1 = self.insightface_app.get(face1)
                faces2 = self.insightface_app.get(face2)
                
                if len(faces1) > 0 and len(faces2) > 0:
                    # Get the first detected face embedding in each image
                    embedding1 = faces1[0].embedding
                    embedding2 = faces2[0].embedding
                    
                    # Calculate cosine similarity
                    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                    
                    # InsightFace similarity threshold
                    is_same = similarity > 0.5  # Adjust threshold as needed
                    return is_same, similarity
            except Exception as e:
                logger.warning(f"InsightFace comparison error: {str(e)}, falling back to other methods")
    
        # Fall back to face_recognition if available
        if HAVE_FACE_RECOGNITION:
            # Use face_recognition for more accurate comparison
            # Convert to RGB (face_recognition expects RGB)
            if face1.shape[2] == 3:
                rgb_face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
                rgb_face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
            else:
                rgb_face1 = face1
                rgb_face2 = face2
                
            # Get face encodings
            try:
                encoding1 = face_recognition.face_encodings(rgb_face1, num_jitters=1)
                encoding2 = face_recognition.face_encodings(rgb_face2, num_jitters=1)
                
                # Check if encodings were found
                if len(encoding1) == 0 or len(encoding2) == 0:
                    return False, 0.0
                    
                # Compare face encodings
                face_distance = face_recognition.face_distance([encoding1[0]], encoding2[0])[0]
                
                # Convert distance to similarity score (distance closer to 0 means more similar)
                similarity = 1.0 - face_distance
                
                # Consider faces similar if similarity is above threshold
                is_same = similarity > self.identical_face_threshold
                return is_same, similarity
            except Exception as e:
                logger.warning(f"Face comparison error: {str(e)}, falling back to basic methods")
        
        # Fall back to simple comparison as last resort
        # Resize both to same size
        face1_resized = cv2.resize(face1, (128, 128))
        face2_resized = cv2.resize(face2, (128, 128))
        
        # Convert to grayscale
        if face1_resized.ndim > 2:
            face1_gray = cv2.cvtColor(face1_resized, cv2.COLOR_BGR2GRAY)
        else:
            face1_gray = face1_resized
            
        if face2_resized.ndim > 2:
            face2_gray = cv2.cvtColor(face2_resized, cv2.COLOR_BGR2GRAY)
        else:
            face2_gray = face2_resized
            
        # Compare using structural similarity
        try:
            from skimage.metrics import structural_similarity as ssim
            score = ssim(face1_gray, face2_gray)
            
            # Consider faces similar if score is above threshold
            is_same = score > 0.7
            return is_same, score
        except:
            # If skimage not available, use normalized correlation
            result = cv2.matchTemplate(face1_gray, face2_gray, cv2.TM_CCORR_NORMED)
            score = result[0][0]
            
            # Normalize score to 0-1 range
            score = (score + 1.0) / 2.0
            
            # Consider faces similar if score is above threshold
            is_same = score > 0.75
            return is_same, score
    
    def find_matching_face(self, face_image, face_list):
        """
        Find a matching face in a list of face images
        
        Args:
            face_image: The face to search for
            face_list: List of face images to compare against
            
        Returns:
            Tuple of (index of matching face or -1, similarity score)
        """
        if face_image is None or not face_list:
            return -1, 0.0
            
        best_match_index = -1
        best_match_score = 0.0
        
        for i, other_face in enumerate(face_list):
            is_match, score = self.compare_face_similarity(face_image, other_face)
            
            if is_match and score > best_match_score:
                best_match_score = score
                best_match_index = i
                
        return best_match_index, best_match_score
    
    def get_facial_landmarks(self, image, face_rect):
        """
        Get facial landmarks (eyes, nose, mouth, etc.) for a detected face
        
        Returns:
            Dictionary of landmark points or None if not available
        """
        if not self.landmarks_predictor or not HAVE_FACE_RECOGNITION:
            return None
            
        try:
            # Convert to RGB (dlib expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert rectangle to dlib format
            x, y, w, h = face_rect
            dlib_rect = dlib.rectangle(x, y, x + w, y + h)
            
            # Get landmarks
            shape = self.landmarks_predictor(rgb_image, dlib_rect)
            
            # Convert to numpy array
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Organize landmarks
            facial_features = {
                "jaw": landmarks[0:17],
                "right_eyebrow": landmarks[17:22],
                "left_eyebrow": landmarks[22:27],
                "nose_bridge": landmarks[27:31],
                "nose_tip": landmarks[31:36],
                "right_eye": landmarks[36:42],
                "left_eye": landmarks[42:48],
                "outer_lip": landmarks[48:60],
                "inner_lip": landmarks[60:68]
            }
            
            return facial_features
        except Exception as e:
            logger.warning(f"Error getting facial landmarks: {str(e)}")
            return None
    
    def enhance_face(self, image, face_rect) -> np.ndarray:
        """
        Advanced face enhancement with multiple techniques for optimal quality
        
        Args:
            image: Input image
            face_rect: Face rectangle (x, y, width, height)
            
        Returns:
            Enhanced face image
        """
        x, y, w, h = face_rect
        
        # Extract face region with dynamic margin based on face size
        # Larger margin for small faces, smaller for large faces
        margin_factor = 0.2 if w < 100 else 0.1
        margin_x = int(w * margin_factor)
        margin_y = int(h * margin_factor)
        
        # Ensure we stay within image bounds
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        # Extract face region
        face_img = image[y1:y2, x1:x2].copy()
        if face_img.size == 0:
            # Fallback to original rect if margins caused issues
            face_img = image[y:y+h, x:x+w].copy()
            if face_img.size == 0:
                logger.warning("Invalid face region after extraction")
                return image[max(0, y):min(image.shape[0], y+h), 
                            max(0, x):min(image.shape[1], x+w)].copy()
        
        # Get landmarks to help with face enhancement
        landmarks = self.get_facial_landmarks(image, face_rect)
        
        # Apply advanced enhancement pipeline
        try:
            # 1. Start with denoising - adapt strength based on image quality
            # Check image quality to determine denoising strength
            quality = self.get_quality_score(face_img)
            
            # More aggressive denoising for lower quality images
            h_value = 15 if quality < 0.4 else 10 if quality < 0.7 else 5
            denoised = cv2.fastNlMeansDenoisingColored(face_img, None, h_value, h_value, 7, 21)
            
            # 2. Apply advanced color correction
            enhanced = self._apply_color_correction(denoised)
            
            # 3. Apply adaptive contrast enhancement
            enhanced = self._adaptive_contrast_enhancement(enhanced)
            
            # 4. Sharpen with dynamic parameters based on image size and quality
            if face_img.shape[0] > 80 and face_img.shape[1] > 80:  # Only sharpen if face is big enough
                # Dynamic sharpening - more for larger faces, less for smaller ones
                sharpening_strength = min(0.4, max(0.1, face_img.shape[0] / 500))
                enhanced = self._adaptive_sharpening(enhanced, strength=sharpening_strength)
            
            # 5. Gamma correction if the face is too dark or too bright
            brightness = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)) / 255.0
            if brightness < 0.4:
                # Image is too dark, brighten it
                gamma = 0.8
                enhanced = self._adjust_gamma(enhanced, gamma=gamma)
            elif brightness > 0.8:
                # Image is too bright, darken it
                gamma = 1.2
                enhanced = self._adjust_gamma(enhanced, gamma=gamma)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Face enhancement failed: {str(e)}, returning original face")
            return face_img
    
    def _apply_color_correction(self, image):
        """Apply color correction to improve face appearance"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Slightly adjust A and B channels for better skin tone
            # This helps with security camera footage that often has color casts
            height, width = a.shape
            for i in range(height):
                for j in range(width):
                    # Move slightly toward neutral skin tone
                    a[i, j] = int(a[i, j] * 0.95 + 128 * 0.05)
                    b[i, j] = int(b[i, j] * 0.95 + 128 * 0.05)
            
            # Merge channels
            corrected_lab = cv2.merge((cl, a, b))
            
            # Convert back to BGR
            return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"Color correction failed: {str(e)}")
            return image
    
    def _adaptive_contrast_enhancement(self, image):
        """Apply adaptive contrast enhancement based on image content"""
        try:
            # Convert to YCrCb color space
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            # Split channels
            y, cr, cb = cv2.split(ycrcb)
            
            # Apply CLAHE to Y channel with adaptive parameters
            # Calculate optimal clip limit based on image histogram
            hist = cv2.calcHist([y], [0], None, [256], [0, 256])
            hist_normalized = hist / (y.shape[0] * y.shape[1])
            hist_std = np.std(hist_normalized)
            
            # Higher clip limit for more uniform histograms
            clip_limit = max(1.5, min(4.0, 2.5 / (hist_std + 0.1)))
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            y_enhanced = clahe.apply(y)
            
            # Merge channels
            ycrcb_enhanced = cv2.merge((y_enhanced, cr, cb))
            
            # Convert back to BGR
            return cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)
        except Exception as e:
            logger.warning(f"Adaptive contrast enhancement failed: {str(e)}")
            return image
    
    def _adaptive_sharpening(self, image, strength=0.3):
        """Apply adaptive sharpening with edge preservation"""
        try:
            # Blur the image slightly
            blurred = cv2.GaussianBlur(image, (0, 0), 3)
            
            # Calculate the difference to get edges
            diff = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            
            # Apply bilateral filter to preserve edges while removing noise
            return cv2.bilateralFilter(diff, 5, 35, 35)
        except Exception as e:
            logger.warning(f"Adaptive sharpening failed: {str(e)}")
            return image
    
    def _adjust_gamma(self, image, gamma=1.0):
        """Adjust gamma of an image to enhance brightness/darkness"""
        try:
            # Build a lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
            ]).astype("uint8")
            
            # Apply gamma correction
            return cv2.LUT(image, table)
        except Exception as e:
            logger.warning(f"Gamma adjustment failed: {str(e)}")
            return image
    
    def get_quality_score(self, face_img) -> float:
        """
        Calculate a robust quality score for a face image (0.0 to 1.0)
        Higher is better quality
        """
        if face_img is None or face_img.size == 0:
            return 0.0
            
        try:
            # Convert to grayscale
            if len(face_img.shape) > 2:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
                
            # Calculate multiple metrics
            
            # 1. Variance of Laplacian (focus measure)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            focus_score = min(1.0, laplacian.var() / 500.0)  # Normalize and cap
            
            # 2. Overall brightness and deviation from ideal
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - 2.0 * abs(0.5 - brightness)  # Penalize too dark or too bright
            
            # 3. Contrast
            min_val, max_val, _, _ = cv2.minMaxLoc(gray)
            contrast_score = min(1.0, (max_val - min_val) / 255.0)
            
            # 4. Entropy (texture information)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / hist.sum()  # Normalize
            non_zero = hist > 0
            entropy = -np.sum(hist[non_zero] * np.log2(hist[non_zero]))
            entropy_score = min(1.0, entropy / 8.0)  # Normalize, theoretical max is 8
            
            # 5. Face size score (bigger is better)
            size_score = min(1.0, (face_img.shape[0] * face_img.shape[1]) / (250 * 250))
            
            # Weighted average based on importance for face recognition
            quality_score = (0.35 * focus_score + 
                           0.15 * brightness_score + 
                           0.20 * contrast_score + 
                           0.15 * entropy_score +
                           0.15 * size_score)
                           
            return min(1.0, max(0.0, quality_score))  # Ensure it's in the 0-1 range
            
        except Exception as e:
            logger.warning(f"Error calculating face quality: {str(e)}")
            return 0.0
    
    def get_best_face(self, frame, faces=None) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
        """
        Detect and return the best quality face in the frame
        
        Args:
            frame: Input frame
            faces: Pre-detected faces or None to detect faces
            
        Returns:
            Tuple of (enhanced_face_image, face_rect, quality_score)
        """
        if frame is None:
            return None, None, 0.0
            
        # Detect faces if not provided
        if faces is None:
            faces = self.detect_faces(frame)
            
        if not faces:
            return None, None, 0.0
            
        best_face = None
        best_score = -1
        best_rect = None
        
        # Find the best quality face
        for face_rect in faces:
            # Extract face from frame
            x, y, w, h = face_rect
            if w < self.min_face_size[0] or h < self.min_face_size[1]:
                continue
                
            # Get face image
            face_img = frame[y:y+h, x:x+w].copy()
            
            # Calculate quality score
            score = self.get_quality_score(face_img)
            
            # Update if better
            if score > best_score:
                best_score = score
                best_rect = face_rect
        
        # If found a good face, enhance it
        if best_rect is not None:
            enhanced_face = self.enhance_face(frame, best_rect)
            return enhanced_face, best_rect, best_score
            
        return None, None, 0.0
    
    def draw_face_rectangles(self, frame, faces, color=(0, 255, 0), thickness=2):
        """
        Draw rectangles around detected faces with quality scores
        
        Args:
            frame: Input frame
            faces: List of face rectangles
            color: Rectangle color (B, G, R)
            thickness: Line thickness
            
        Returns:
            Frame with face rectangles drawn
        """
        result = frame.copy()
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle
            cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
            
            # Calculate and show quality score
            face_img = frame[y:y+h, x:x+w]
            quality = self.get_quality_score(face_img)
            
            # Display quality score with background for visibility
            text = f"Q: {quality:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw text background
            cv2.rectangle(result, 
                        (x, y - text_size[1] - 5), 
                        (x + text_size[0], y), 
                        (0, 0, 0), 
                        -1)
            
            # Draw text
            cv2.putText(result, text, (x, y - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result