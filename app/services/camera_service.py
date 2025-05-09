import json
import os
import uuid
from typing import Optional, Dict, List, Any
from datetime import datetime
from app.core.config import settings
from app.models.camera import Camera, CameraCreate, FilterConfig # Ensure FilterConfig is imported if used directly
from app.utils.stream_validator import StreamValidator
from app.core.websocket_manager import manager # Added for WebSocket broadcasting
import asyncio # Added for running async broadcast

class CameraService:
    def __init__(self):
        # Ensure the data directory exists
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        self.cameras_file = settings.DATA_DIR / "cameras.json"
        
        # Initialize cameras.json if it doesn't exist
        if not os.path.exists(self.cameras_file):
            with open(self.cameras_file, "w") as f:
                json.dump({}, f)
            print(f"Initialized empty cameras file at {self.cameras_file}")
    
    def _read_cameras(self) -> Dict:
        """Read cameras from JSON file"""
        cameras_data = {}
        try:
            with open(self.cameras_file, "r") as f:
                loaded_data = json.load(f)
                for cam_id, cam_details in loaded_data.items():
                    if "is_active" not in cam_details:
                        cam_details["is_active"] = False # Default for old records
                    cameras_data[cam_id] = cam_details
                return cameras_data
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _write_cameras(self, cameras_data: Dict) -> None:
        """Write cameras to JSON file"""
        with open(self.cameras_file, "w") as f:
            json.dump(cameras_data, f, indent=4)
        print(f"Saved camera data to {self.cameras_file}")
    
    def create_or_update_camera(self, camera_data: CameraCreate, validation_result: Optional[Dict[str, Any]] = None) -> Camera:
        """Create or update a camera with optional validation results"""
        # Read existing cameras
        cameras = self._read_cameras()
        
        # Generate a unique camera ID if it's a new camera or use existing ID
        camera_id = None
        # Check if camera with same name already exists
        for cam_id, cam_info in cameras.items():
            if cam_info["name"] == camera_data.name:
                camera_id = cam_id
                break
        
        if camera_id:
            # Update existing camera
            cameras[camera_id]["rtsp_url"] = camera_data.rtsp_url
            cameras[camera_id]["updated_at"] = datetime.now().isoformat()
            cameras[camera_id]["filters"] = [filter_config.dict() for filter_config in camera_data.filters] if camera_data.filters else []
            
            # Add validation results if provided
            if validation_result:
                cameras[camera_id]["stream_status"] = validation_result["is_valid"]
                cameras[camera_id]["validation_result"] = validation_result
                cameras[camera_id]["is_active"] = validation_result["is_valid"] # Update is_active status
        else:
            # Create new camera
            camera_id = str(uuid.uuid4())
            
            # Prepare camera data
            camera_dict = {
                "id": camera_id,
                "name": camera_data.name,
                "rtsp_url": camera_data.rtsp_url,
                "created_at": camera_data.created_at,
                "updated_at": None,
                "filters": [filter_config.dict() for filter_config in camera_data.filters] if camera_data.filters else [],
                "is_active": False # Default for new camera, to be updated by validation/stream
            }
            
            # Add validation results if provided
            if validation_result:
                camera_dict["stream_status"] = validation_result["is_valid"]
                camera_dict["validation_result"] = validation_result
                camera_dict["is_active"] = validation_result["is_valid"] # Set is_active based on validation
            
            cameras[camera_id] = camera_dict
        
        # Save changes
        self._write_cameras(cameras)
        
        # Before returning, ensure the camera object reflects the data just written
        final_camera_data = cameras[camera_id]
        return Camera(**final_camera_data)
    
    def get_camera(self, camera_id: str) -> Optional[Camera]:
        """Get a camera by ID"""
        cameras = self._read_cameras()
        if camera_id not in cameras:
            return None
        
        return Camera(**cameras[camera_id])
    
    def get_camera_by_id(self, camera_id: str) -> Optional[Camera]:
        """Get a camera by ID - alias for get_camera"""
        return self.get_camera(camera_id)
    
    def get_camera_by_name(self, name: str) -> Optional[Camera]:
        """Get a camera by name"""
        cameras = self._read_cameras()
        for camera_id, camera_info in cameras.items():
            if camera_info["name"] == name:
                return Camera(**camera_info)
        
        return None
    
    def update_camera_filters(self, camera_id: str, filters: List[Dict[str, Any]]) -> Optional[Camera]:
        """Update a camera's filters"""
        cameras = self._read_cameras()
        if camera_id not in cameras:
            return None
        
        # Validate that each filter has required fields
        validated_filters = []
        for filter_data in filters:
            # Check if filter has required fields
            if "filter_id" not in filter_data or "filter_name" not in filter_data:
                continue
            
            # Create a validated filter with only the required fields
            validated_filter = {
                "filter_id": filter_data["filter_id"],
                "filter_name": filter_data["filter_name"],
                "enabled": filter_data.get("enabled", True)  # Default to True if not provided
            }
            validated_filters.append(validated_filter)
        
        # Update the camera with validated filters
        cameras[camera_id]["filters"] = validated_filters
        cameras[camera_id]["updated_at"] = datetime.now().isoformat()
        
        # Save changes
        self._write_cameras(cameras)
        
        # Return updated camera
        return Camera(**cameras[camera_id])
    
    def delete_camera(self, camera_id: str) -> bool:
        """Delete a camera by ID"""
        cameras = self._read_cameras()
        if camera_id not in cameras:
            return False
        
        # Delete camera from cameras.json
        del cameras[camera_id]
        self._write_cameras(cameras)
        
        return True
    
    def delete_camera_by_name(self, name: str) -> bool:
        """Delete a camera by name"""
        cameras = self._read_cameras()
        camera_id = None
        
        for cam_id, cam_info in cameras.items():
            if cam_info["name"] == name:
                camera_id = cam_id
                break
        
        if not camera_id:
            return False
        
        # Delete camera from cameras.json
        del cameras[camera_id]
        self._write_cameras(cameras)
        
        return True
    
    def list_cameras(self) -> List[Camera]:
        """List all cameras"""
        cameras = self._read_cameras()
        return [Camera(**camera_data) for camera_data in cameras.values()]
        
    def get_all_cameras(self) -> Dict[str, Camera]:
        """Get all cameras as a dictionary with camera ID as key"""
        cameras_dict = {}
        cameras = self._read_cameras()
        for camera_id, camera_data in cameras.items():
            cameras_dict[camera_id] = Camera(**camera_data)
        return cameras_dict
        
    async def update_camera_active_status(self, camera_id: str, is_active: bool) -> Optional[Camera]:
        """Update the active status of a camera and broadcast the change."""
        cameras = self._read_cameras()
        if camera_id not in cameras:
            return None

        if cameras[camera_id].get("is_active") == is_active: # No change
            return Camera(**cameras[camera_id])

        cameras[camera_id]["is_active"] = is_active
        cameras[camera_id]["updated_at"] = datetime.now().isoformat()
        self._write_cameras(cameras)

        notification_payload = {
            "type": "camera_status_update",
            "camera_id": camera_id,
            "name": cameras[camera_id].get("name", "Unknown Camera"),
            "is_active": is_active,
            "timestamp": datetime.now().isoformat()
        }
        # Run broadcast in a new task to avoid blocking if called from sync code,
        # or await if called from async. Since this service might be called from sync/async,
        # creating a task is safer.
        asyncio.create_task(manager.broadcast_json(notification_payload))
        
        return Camera(**cameras[camera_id])

    def validate_camera_stream(self, camera_id: str) -> Optional[Dict]:
        """Validate an existing camera's stream"""
        camera = self.get_camera(camera_id)
        if not camera:
            return None
            
        # Validate the stream using the new interface
        validation_result = StreamValidator.validate_rtsp_stream(camera.rtsp_url)
        
        # Update the camera with validation results
        cameras = self._read_cameras()
        cameras[camera_id]["stream_status"] = validation_result["is_valid"]
        cameras[camera_id]["last_validation"] = validation_result
        cameras[camera_id]["updated_at"] = datetime.now().isoformat()
        cameras[camera_id]["is_active"] = validation_result["is_valid"] # Update is_active status

        self._write_cameras(cameras)
        
        # Broadcast status change after validation
        # This might be redundant if AI engine also calls update_camera_active_status
        # Consider if validation itself should trigger a broadcast or if it's a separate concern.
        # For now, let's assume AI engine will be the primary source of active status updates.
        # However, if validation is a manual trigger, broadcasting here is useful.
        asyncio.create_task(self.update_camera_active_status(camera_id, validation_result["is_valid"]))

        return validation_result

# Create a singleton instance for global use
camera_service = CameraService()

# Module level function that calls the instance method
def get_camera_by_id(camera_id: str) -> Optional[Camera]:
    """Get a camera by ID - module level function that calls the CameraService instance"""
    return camera_service.get_camera(camera_id)

# Add the missing get_all_cameras function at module level
def get_all_cameras() -> Dict[str, Camera]:
    """Get all cameras as a dictionary with camera ID as key - module level function"""
    return camera_service.get_all_cameras()