from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

# Use completely flexible model for RuleCondition
class RuleCondition(BaseModel):
    type: str
    data: Dict[str, Any]  # This allows any structure in the data field
    
    class Config:
        extra = "allow"  # Allow extra fields not defined in the model

class Rule(BaseModel):
    id: Optional[str] = None
    name: str
    event: str
    condition: RuleCondition
    enabled: bool = True
    days: Optional[List[str]] = []  # Make days optional with default empty list
    cameraId: str
    cameraName: str
    created_at: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True
        extra = "allow"  # Allow extra fields
        
class RuleCreate(BaseModel):
    name: str
    event: str
    condition: RuleCondition
    enabled: bool = True
    days: Optional[List[str]] = []  # Make days optional with default empty list
    cameraId: str
    cameraName: str
    
    class Config:
        extra = "allow"  # Allow extra fields that aren't defined in the model
    
class RuleUpdate(BaseModel):
    name: Optional[str] = None
    event: Optional[str] = None
    condition: Optional[RuleCondition] = None
    enabled: Optional[bool] = None
    days: Optional[List[str]] = None
    cameraId: Optional[str] = None
    cameraName: Optional[str] = None

    class Config:
        extra = "allow"  # Allow extra fields

class RuleResponse(BaseModel):
    id: str
    name: str
    event: str
    condition: RuleCondition
    enabled: bool
    days: List[str] = []  # Default empty list
    cameraId: str
    cameraName: str
    created_at: str
    updated_at: Optional[str] = None
    message: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow extra fields