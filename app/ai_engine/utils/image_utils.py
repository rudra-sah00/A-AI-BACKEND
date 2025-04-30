import cv2
import numpy as np
import logging
import os
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

def resize_image(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        width: Target width (if None, will be calculated from height)
        height: Target height (if None, will be calculated from width)
        
    Returns:
        Resized image
    """
    if width is None and height is None:
        return image
    
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    if width is None:
        aspect = height / float(h)
        width = int(w * aspect)
    elif height is None:
        aspect = width / float(w)
        height = int(h * aspect)
        
    # Resize and return
    return cv2.resize(image, (width, height))

def draw_bounding_box(image: np.ndarray, bbox: List[int], label: str = '', 
                     color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw a bounding box on an image
    
    Args:
        image: Input image
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        label: Label text
        color: Box color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with bounding box
    """
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size = cv2.getTextSize(label, font, font_scale, 1)[0]
        text_x = x1
        text_y = y1 - 5 if y1 - 5 > text_size[1] else y1 + text_size[1]
        
        # Draw text background
        cv2.rectangle(image, (text_x, text_y - text_size[1]), 
                    (text_x + text_size[0], text_y), color, -1)
        
        # Draw text
        cv2.putText(image, label, (text_x, text_y), font, font_scale, 
                  (255, 255, 255), 1, cv2.LINE_AA)
    
    return image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to 0-1 range
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image or None if failed
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
            
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None