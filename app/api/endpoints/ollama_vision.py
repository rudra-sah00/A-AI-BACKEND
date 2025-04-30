import logging
from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from app.core.config import settings
from app.services.camera_service import camera_service
from app.ai_engine.engine import AIEngine

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
async def process_contextual_query(request: QueryRequest, ai_engine: AIEngine = Depends(lambda: settings.ai_engine)):
    """
    Process a contextual query for a camera using the OllamaVision model
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
        
        # Check if the camera has the OllamaVision filter enabled
        ollama_vision_enabled = False
        for filter_config in camera.get("filters", []):
            if filter_config.get("filter_name") == "OllamaVision" and filter_config.get("enabled", False):
                ollama_vision_enabled = True
                break
        
        if not ollama_vision_enabled:
            logger.error(f"OllamaVision filter not enabled for camera: {request.camera_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"OllamaVision filter not enabled for camera: {request.camera_id}"
            )
        
        # Get or create the OllamaVision processor for this camera
        processor_key = f"ollamavision_{request.camera_id}"
        
        if processor_key not in ai_engine.active_processors:
            # Import here to avoid circular imports
            from app.ai_engine.processors.ollama_vision_processor import OllamaVisionProcessor
            
            # Create the output directory
            output_dir = settings.DATA_DIR / "unauthorized"
            
            # Create the processor
            logger.info(f"Creating OllamaVision processor for camera {camera.get('name')} (ID: {request.camera_id})")
            processor = OllamaVisionProcessor(
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
            logger.error(f"OllamaVision processor for camera {request.camera_id} is not active")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"OllamaVision processor for camera {request.camera_id} is not active. Please wait for system to initialize."
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
async def get_ollamavision_cameras(ai_engine: AIEngine = Depends(lambda: settings.ai_engine)):
    """
    Get all cameras with OllamaVision filter enabled
    """
    try:
        # Get all cameras
        cameras = camera_service.get_all_cameras()
        
        # Filter cameras with OllamaVision enabled
        ollamavision_cameras = []
        for camera_id, camera in cameras.items():
            for filter_config in camera.get("filters", []):
                if filter_config.get("filter_name") == "OllamaVision" and filter_config.get("enabled", False):
                    # Check if the processor is active
                    processor_key = f"ollamavision_{camera_id}"
                    is_active = processor_key in ai_engine.active_processors and ai_engine.active_processors[processor_key].is_active
                    
                    # Add camera with active status
                    camera_info = {
                        "id": camera_id,
                        "name": camera.get("name"),
                        "is_active": is_active
                    }
                    ollamavision_cameras.append(camera_info)
                    break
        
        return ollamavision_cameras
        
    except Exception as e:
        logger.error(f"Error getting OllamaVision cameras: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting OllamaVision cameras: {str(e)}"
        )