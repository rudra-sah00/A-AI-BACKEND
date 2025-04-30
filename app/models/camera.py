from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class FilterConfig(BaseModel):
    filter_id: str
    filter_name: str
    enabled: bool = True
    # Removed config field to simplify


class CameraBase(BaseModel):
    name: str
    rtsp_url: str
    filters: Optional[List[FilterConfig]] = []


class CameraCreate(CameraBase):
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class Camera(CameraBase):
    id: str
    created_at: str
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class CameraResponse(BaseModel):
    id: Optional[str] = None
    name: str
    rtsp_url: str
    filters: Optional[List[FilterConfig]] = []
    message: str


class ContextualQueryRequest(BaseModel):
    camera_id: str
    query: str
    priority: Optional[bool] = False
    
    
class ContextualQueryResponse(BaseModel):
    query_id: str
    status: str
    response: str
    camera_id: str
    timestamp: float
    image_base64: Optional[str] = None