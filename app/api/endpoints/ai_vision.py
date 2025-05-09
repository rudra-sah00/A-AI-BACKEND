import logging
from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from app.core.config import settings
from app.services.camera_service import camera_service
from app.ai_engine.engine import AIEngine
from app.ai_engine import ai_engine  # Import the ai_engine instance directly

logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    camera_id: str
    query: str

class QueryResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    timestamp: Optional[str] = None
    model: Optional[str] = None

@router.post("/query", response_model=QueryResponse)
async def process_contextual_query(request: QueryRequest):
    """
    Process a contextual query for a camera using the Gemini Flash 2.0 vision model
    """
    try:
        # Check if the camera exists
        camera = camera_service.get_camera(request.camera_id)
        if not camera:
            logger.error(f"Camera not found: {request.camera_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera not found: {request.camera_id}"
            )
        
        # Check if the camera has the Vision filter enabled
        # Note: We're still using "OllamaVision" as the filter name for compatibility
        # but internally it's now using Gemini Flash 2.0
        vision_filter_enabled = False
        for filter_config in camera.filters:
            if filter_config.filter_name == "OllamaVision" and filter_config.enabled:
                vision_filter_enabled = True
                break
        
        if not vision_filter_enabled:
            logger.error(f"Vision filter not enabled for camera: {request.camera_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Vision filter not enabled for camera: {request.camera_id}"
            )
        
        # Get or create the Vision processor for this camera
        processor_key = f"ollamavision_{request.camera_id}"
        
        if processor_key not in ai_engine.active_processors:
            # Import here to avoid circular imports
            from app.ai_engine.processors.ai_vision_processor import OllamaVisionProcessor
            
            # Create the output directory
            output_dir = settings.DATA_DIR / "unauthorized"
            
            # Create the processor
            logger.info(f"Creating contextual query processor for camera {camera.name} (ID: {request.camera_id})")
            
            # Convert Camera object to dictionary for the processor
            # Use model_dump() instead of dict() for Pydantic v2 compatibility
            try:
                # First try model_dump() (Pydantic v2)
                filters_list = [filter_config.model_dump() for filter_config in camera.filters]
            except AttributeError:
                try:
                    # Then try dict() (Pydantic v1)
                    filters_list = [filter_config.dict() for filter_config in camera.filters]
                except AttributeError:
                    # Fallback to manual dictionary creation
                    filters_list = []
                    for filter_config in camera.filters:
                        filters_list.append({
                            "filter_id": filter_config.filter_id,
                            "filter_name": filter_config.filter_name,
                            "enabled": filter_config.enabled
                        })
            
            camera_dict = {
                "id": camera.id,
                "name": camera.name,
                "rtsp_url": camera.rtsp_url,
                "filters": filters_list,
                "is_active": camera.is_active
            }
            
            processor = OllamaVisionProcessor(
                camera_dict,
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
            logger.error(f"Vision processor for camera {request.camera_id} is not active")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Vision processor for camera {request.camera_id} is not active. Please wait for system to initialize."
            )
        
        # Process the query
        result = processor.process_query(request.query)
        
        if not result.get("success", False):
            logger.error(f"Error processing contextual query: {result.get('error')}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("error", "Unknown error processing query")
            )
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing contextual query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing contextual query: {str(e)}"
        )

@router.get("/cameras", response_model=List[Dict])
async def get_vision_cameras():
    """
    Get all cameras with contextual vision capability enabled
    """
    try:
        # Get all cameras
        cameras_dict = camera_service.get_all_cameras()
        
        # Filter cameras with Vision enabled
        vision_cameras = []
        for camera_id, camera in cameras_dict.items():
            vision_filter_enabled = False
            for filter_config in camera.filters:
                if filter_config.filter_name == "OllamaVision" and filter_config.enabled:
                    vision_filter_enabled = True
                    break
                    
            if vision_filter_enabled:
                # Check if the processor is active
                processor_key = f"ollamavision_{camera_id}"
                is_active = processor_key in ai_engine.active_processors and ai_engine.active_processors[processor_key].is_active
                
                # Add camera with active status
                camera_info = {
                    "id": camera.id,
                    "name": camera.name,
                    "is_active": is_active,
                    "model": "gemini-2.0-flash"  # Add the model information
                }
                vision_cameras.append(camera_info)
        
        return vision_cameras
        
    except Exception as e:
        logger.error(f"Error getting contextual vision cameras: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting contextual vision cameras: {str(e)}"
        )