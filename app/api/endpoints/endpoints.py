import logging
import os
import shutil
import tempfile
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime

from app.core.config import settings
from app.services.camera_service import camera_service
from app.ai_engine.engine import AIEngine
from app.ai_engine.processors.gunny_bag_processor import GunnyBagProcessor

logger = logging.getLogger(__name__)

router = APIRouter()

class GunnyBagCountResponse(BaseModel):
    success: bool
    count: Optional[int] = None
    avg_count: Optional[float] = None
    max_count: Optional[int] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None
    image_url: Optional[str] = None
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    error: Optional[str] = None
    
class VehicleRecognizeResponse(BaseModel):
    success: bool
    vehicle_type: Optional[str] = None
    license_plate: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: Optional[str] = None
    image_url: Optional[str] = None
    error: Optional[str] = None

@router.post("/gunny-bag-count", response_model=GunnyBagCountResponse)
async def count_gunny_bags(
    camera_id: Optional[str] = Query(None, description="Camera ID to use for counting"),
    ai_engine: AIEngine = Depends(lambda: settings.ai_engine)
):
    """
    Count gunny bags in real-time from a camera feed
    """
    try:
        # Check if a camera ID was provided
        if not camera_id:
            # Use the first available camera if none specified
            cameras = camera_service.get_all_cameras()
            if not cameras:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No cameras configured in the system"
                )
            camera_id = list(cameras.keys())[0]
        
        # Get the camera
        camera = camera_service.get_camera(camera_id)
        if not camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera not found: {camera_id}"
            )
        
        # Check if the camera is available
        if not camera.get("stream_status", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Camera {camera.get('name')} is not available"
            )
        
        # Get or create a gunny bag processor for this camera
        processor_key = f"gunny_bag_{camera_id}"
        
        if processor_key not in ai_engine.active_processors:
            # Create the output directory
            output_dir = settings.DATA_DIR / "unauthorized"  # Reusing existing directory for test purposes
            
            # Create the processor
            logger.info(f"Creating GunnyBagProcessor for camera {camera.get('name')} (ID: {camera_id})")
            processor = GunnyBagProcessor(
                camera,
                ai_engine.users,
                str(output_dir)
            )
            
            # Start the processor
            processor.process()
            
            # Add to active processors
            ai_engine.active_processors[processor_key] = processor
        
        # Get the processor
        processor = ai_engine.active_processors.get(processor_key)
        
        if not processor or not processor.is_active:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"GunnyBagProcessor for camera {camera_id} is not active"
            )
        
        # Get the latest result
        result = processor.get_latest_result()
        
        # Check if we have a result
        if result["timestamp"] is None:
            return GunnyBagCountResponse(
                success=True,
                count=0,
                confidence=0.0,
                camera_id=camera_id,
                camera_name=camera.get("name"),
                timestamp=datetime.now().isoformat(),
                error="No detection results yet, processing started"
            )
        
        # Convert the image path to a URL
        image_url = None
        if result.get("image_path") and os.path.exists(result["image_path"]):
            # For a real API, you would serve this through a static file endpoint
            # Here we just return the file path
            image_url = f"/static/gunny_bags/{os.path.basename(result['image_path'])}"
        
        return GunnyBagCountResponse(
            success=True,
            count=result["count"],
            confidence=result["confidence"],
            timestamp=datetime.fromtimestamp(result["timestamp"]).isoformat(),
            image_url=image_url,
            camera_id=camera_id,
            camera_name=camera.get("name")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error counting gunny bags: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error counting gunny bags: {str(e)}"
        )

@router.post("/gunny-bag-count/upload", response_model=GunnyBagCountResponse)
async def count_gunny_bags_in_video(
    video: UploadFile = File(...),
    ai_engine: AIEngine = Depends(lambda: settings.ai_engine)
):
    """
    Count gunny bags in an uploaded video file
    """
    temp_file = None
    try:
        # Save the uploaded file to a temporary location
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video.filename}")
        
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Create a gunny bag processor
        output_dir = settings.DATA_DIR / "unauthorized"  # Reusing existing directory for test purposes
        
        # Use a dummy camera for processing videos
        dummy_camera = {
            "id": "video_upload",
            "name": "Video Upload"
        }
        
        # Create a processor just for this video
        processor = GunnyBagProcessor(
            dummy_camera,
            ai_engine.users,
            str(output_dir)
        )
        
        # Process the video
        result = processor.process_video(temp_file)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Unknown error processing video")
            )
        
        # Convert the image path to a URL if available
        image_url = None
        if result.get("result_image") and os.path.exists(result["result_image"]):
            image_url = f"/static/gunny_bags/{os.path.basename(result['result_image'])}"
        
        return GunnyBagCountResponse(
            success=True,
            count=int(result.get("max_count", 0)),
            avg_count=result.get("avg_count", 0),
            max_count=result.get("max_count", 0),
            timestamp=result.get("timestamp"),
            confidence=0.7,  # Mock confidence value
            image_url=image_url
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded video: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing uploaded video: {str(e)}"
        )
    finally:
        # Clean up the temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

@router.post("/vehicle-recognize", response_model=VehicleRecognizeResponse)
async def recognize_vehicle(
    camera_id: Optional[str] = Query(None, description="Camera ID to use for vehicle recognition"),
    ai_engine: AIEngine = Depends(lambda: settings.ai_engine)
):
    """
    Recognize vehicles in real-time from a camera feed
    This is a placeholder endpoint for testing purposes
    """
    # Simple mock response since we don't have the actual implementation
    return VehicleRecognizeResponse(
        success=True,
        vehicle_type="Truck",
        license_plate="AB123CD",
        confidence=0.85,
        timestamp=datetime.now().isoformat(),
        image_url=None
    )

@router.post("/vehicle-recognize/upload", response_model=VehicleRecognizeResponse)
async def recognize_vehicle_in_image(
    image: UploadFile = File(...)
):
    """
    Recognize vehicles in an uploaded image
    This is a placeholder endpoint for testing purposes
    """
    # Simple mock response since we don't have the actual implementation
    return VehicleRecognizeResponse(
        success=True,
        vehicle_type="Car",
        license_plate="XY456Z",
        confidence=0.92,
        timestamp=datetime.now().isoformat(),
        image_url=None
    )