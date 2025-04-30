from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class UserProfileBase(BaseModel):
    username: str
    full_name: str  # Added full_name field
    age: int
    role: str


class UserProfileCreate(UserProfileBase):
    photo_path: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class UserProfile(UserProfileBase):
    id: str
    photo_path: Optional[str] = None
    photo_url: Optional[str] = None  # Added photo_url field
    created_at: str
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True  # Updated from orm_mode to from_attributes for Pydantic v2


class UserProfileResponse(UserProfileBase):
    photo_url: Optional[str] = None
    message: str