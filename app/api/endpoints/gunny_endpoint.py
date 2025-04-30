import os
import shutil
import tempfile
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from app.core.config import settings
from app.ai_engine.processors.gunny_bag_counter import GunnyBagCounter

logger = logging.getLogger(__name__)
router = APIRouter()

class GunnyBagCountResult(BaseModel):
    success: bool
    count: int = 0
    avg_count: Optional[float] = None
    timestamp: Optional[str] = None
    image_url: Optional[str] = None
    error: Optional[str] = None

@router.post("/count-gunny-bags", response_model=GunnyBagCountResult)
async def count_gunny_bags(video: UploadFile = File(...)):
    """
    Count gunny bags in uploaded video using YOLOv8
    """
    temp_file = None
    try:
        # Create output directory
        output_dir = os.path.join(settings.DATA_DIR, "gunny_bags")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save uploaded file to temp location
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video.filename}")
        
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Initialize the counter with output directory
        counter = GunnyBagCounter(output_dir=output_dir)
        
        # Process the video
        result = counter.count_in_video(temp_file)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Failed to process video")
            )
        
        # Create image URL if available
        image_url = None
        if result.get("result_image") and os.path.exists(result.get("result_image")):
            # In a real API, you would serve this through a static files endpoint
            image_url = f"/static/gunny_bags/{os.path.basename(result['result_image'])}"
        
        return GunnyBagCountResult(
            success=True,
            count=result.get("count", 0),
            avg_count=result.get("avg_count"),
            timestamp=result.get("timestamp"),
            image_url=image_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error counting gunny bags: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error counting gunny bags: {str(e)}"
        )
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass