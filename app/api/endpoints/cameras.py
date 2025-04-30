from fastapi import APIRouter, HTTPException, Form, Depends, Query, Body
from typing import List, Optional
from datetime import datetime
import json

from app.models.camera import Camera, CameraCreate, CameraResponse, FilterConfig
from app.services.camera_service import CameraService
from app.utils.stream_validator import StreamValidator

router = APIRouter()
camera_service = CameraService()
stream_validator = StreamValidator()


@router.post("/", response_model=CameraResponse)
async def create_camera(
    name: str = Form(...),
    rtsp_url: str = Form(...),
    verify_stream: bool = Form(True)  # Renamed from 'validate' to 'verify_stream'
):
    """
    Create a camera with optional stream validation.
    The RTSP URL should contain embedded authentication credentials if needed (rtsp://user:pass@host/path).
    
    - Set verify_stream=true to verify the RTSP stream is accessible before adding
    - Set verify_stream=false to add the camera without validation
    """
    try:
        # Create camera data object with validation results
        camera_data = CameraCreate(
            name=name,
            rtsp_url=rtsp_url,
            created_at=datetime.now().isoformat()
        )
        
        # Validate the stream if requested
        validation_result = None
        if verify_stream:  # Use the new parameter name
            validation_result = StreamValidator.validate_rtsp_stream(rtsp_url)
            
            if not validation_result["is_valid"]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"RTSP stream validation failed: {validation_result['message']}"
                )
                
            # Save camera with validation data
            camera = camera_service.create_or_update_camera(
                camera_data, 
                validation_result=validation_result
            )
        else:
            # Save camera without validation
            camera = camera_service.create_or_update_camera(camera_data)
        
        # Prepare success message
        message = "Camera added successfully"
        if verify_stream:  # Use the new parameter name
            message += " (stream validated)"
        
        return CameraResponse(
            id=camera.id,
            name=camera.name,
            rtsp_url=camera.rtsp_url,
            message=message
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating camera: {str(e)}")


@router.post("/validate", response_model=dict)
async def validate_stream(
    rtsp_url: str = Form(...),
    timeout: int = Form(5)
):
    """
    Validate an RTSP stream URL without adding it as a camera.
    Useful for testing stream connectivity before adding a camera.
    """
    try:
        validation_result = StreamValidator.validate_rtsp_stream(rtsp_url, timeout)
        
        return {
            "rtsp_url": rtsp_url,
            "is_valid": validation_result["is_valid"],
            "message": validation_result["message"],
            "details": {
                "frame_width": validation_result["frame_width"],
                "frame_height": validation_result["frame_height"],
                "response_time": validation_result["response_time"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating stream: {str(e)}")


@router.get("/", response_model=List[CameraResponse])
async def get_all_cameras():
    """Get all cameras with their filter details"""
    try:
        cameras = camera_service.list_cameras()
        
        return [
            CameraResponse(
                id=camera.id,
                name=camera.name,
                rtsp_url=camera.rtsp_url,
                filters=camera.filters,  # Include the filter details
                message="Camera retrieved successfully"
            ) for camera in cameras
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cameras: {str(e)}")


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str):
    """Get a camera by ID"""
    try:
        camera = camera_service.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera with ID {camera_id} not found")
        
        return CameraResponse(
            id=camera.id,
            name=camera.name,
            rtsp_url=camera.rtsp_url,
            message="Camera retrieved successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving camera: {str(e)}")


@router.delete("/{camera_id}", response_model=dict)
async def delete_camera(camera_id: str):
    """Delete a camera by ID"""
    try:
        success = camera_service.delete_camera(camera_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Camera with ID {camera_id} not found")
        
        return {
            "camera_id": camera_id,
            "message": "Camera deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting camera: {str(e)}")


@router.post("/with-filters", response_model=CameraResponse)
async def create_camera_with_filters(
    name: str = Form(...),
    rtsp_url: str = Form(...),
    filters: str = Form("[]"),  # JSON string of filters
    verify_stream: bool = Form(True)  # Renamed from 'validate' to 'verify_stream'
):
    """
    Create a camera with filters and optional stream validation.
    
    - filters should be a JSON string representing an array of filter configurations
    - Each filter requires: filter_id, filter_name, enabled, and optional config object
    """
    try:
        # Parse the filters JSON string
        try:
            filter_list = json.loads(filters)
            filter_configs = [FilterConfig(**filter_data) for filter_data in filter_list]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid filter configuration: {str(e)}"
            )
        
        # Create camera data object with filters
        camera_data = CameraCreate(
            name=name,
            rtsp_url=rtsp_url,
            filters=filter_configs,
            created_at=datetime.now().isoformat()
        )
        
        # Validate the stream if requested
        validation_result = None
        if verify_stream:  # Use the new parameter name
            validation_result = StreamValidator.validate_rtsp_stream(rtsp_url)
            
            if not validation_result["is_valid"]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"RTSP stream validation failed: {validation_result['message']}"
                )
                
            # Save camera with validation data
            camera = camera_service.create_or_update_camera(
                camera_data, 
                validation_result=validation_result
            )
        else:
            # Save camera without validation
            camera = camera_service.create_or_update_camera(camera_data)
        
        # Prepare success message
        message = "Camera added successfully with filters"
        if verify_stream:  # Use the new parameter name
            message += " (stream validated)"
        
        return CameraResponse(
            id=camera.id,
            name=camera.name,
            rtsp_url=camera.rtsp_url,
            filters=camera.filters,
            message=message
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating camera with filters: {str(e)}")


@router.put("/{camera_id}/filters", response_model=CameraResponse)
async def update_camera_filters(
    camera_id: str,
    filters: List[dict] = Body(...)
):
    """
    Update filters for an existing camera.
    
    Each filter should include:
    - filter_id: A unique ID for the filter
    - filter_name: Name for the filter
    - config: Optional configuration parameters
    - enabled: Whether the filter is enabled (boolean)
    """
    try:
        # Check if camera exists
        camera = camera_service.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera with ID {camera_id} not found")
        
        # Update the filters
        updated_camera = camera_service.update_camera_filters(camera_id, filters)
        
        return CameraResponse(
            id=updated_camera.id,
            name=updated_camera.name,
            rtsp_url=updated_camera.rtsp_url,
            filters=updated_camera.filters,
            message="Camera filters updated successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating camera filters: {str(e)}")


@router.get("/{camera_id}/filters", response_model=List[FilterConfig])
async def get_camera_filters(camera_id: str):
    """Get all filters for a specific camera"""
    try:
        camera = camera_service.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera with ID {camera_id} not found")
        
        return camera.filters
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving camera filters: {str(e)}")