from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
import shutil
import os
from typing import Optional, List
import json
from datetime import datetime
from pathlib import Path
import uuid

from app.core.config import settings
from app.models.user import UserProfile, UserProfileCreate, UserProfileResponse
from app.services.user_service import UserService

router = APIRouter()
user_service = UserService()

@router.post("/profile", response_model=UserProfileResponse)
async def create_user_profile(
    username: str = Form(...),
    full_name: str = Form(...),  # Added full_name parameter
    age: int = Form(...),
    role: str = Form(...),
    photo: Optional[UploadFile] = File(None)
):
    """
    Create or update a user profile with optional photo upload.
    The photo and user data will be saved locally.
    """
    try:
        # Handle photo upload if provided
        photo_path = None
        if photo:
            # Validate file type
            if photo.content_type not in settings.ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Only {', '.join(settings.ALLOWED_IMAGE_TYPES)} are allowed"
                )
            
            # Generate a unique ID for this upload
            unique_id = str(uuid.uuid4())
            
            # Create directory structure data/uniqueid/
            unique_dir = settings.DATA_DIR / unique_id
            os.makedirs(unique_dir, exist_ok=True)
            
            # Get file extension
            file_extension = os.path.splitext(photo.filename)[1]
            
            # Use username as the filename with the original extension
            filename = f"{username}{file_extension}"
            photo_path = str(unique_dir / filename)
            
            # Save the file
            with open(photo_path, "wb") as buffer:
                shutil.copyfileobj(photo.file, buffer)
        
        # Create user profile
        user_data = UserProfileCreate(
            username=username,
            full_name=full_name,  # Added full_name
            age=age,
            role=role,
            photo_path=photo_path,
            created_at=datetime.now().isoformat()
        )
        
        # Get the full user profile with photo_url from the service
        user_profile = user_service.create_or_update_profile(user_data)
        
        return UserProfileResponse(
            username=user_profile.username,
            full_name=user_profile.full_name,  # Added full_name to response
            age=user_profile.age,
            role=user_profile.role,
            photo_url=user_profile.photo_url,
            message="User profile created successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user profile: {str(e)}")


@router.get("/profile/{username}", response_model=UserProfileResponse)
async def get_user_profile(username: str):
    """Get a user profile by username"""
    try:
        user_profile = user_service.get_profile(username)
        if not user_profile:
            raise HTTPException(status_code=404, detail=f"User {username} not found")
        
        return UserProfileResponse(
            username=user_profile.username,
            full_name=user_profile.full_name,  # Added full_name to response
            age=user_profile.age,
            role=user_profile.role,
            photo_url=user_profile.photo_url,
            message="User profile retrieved successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user profile: {str(e)}")


@router.get("/profiles", response_model=List[UserProfileResponse])
async def get_all_user_profiles():
    """Get all user profiles from the database"""
    try:
        user_profiles = user_service.list_profiles()
        
        # Convert to response models using saved photo_url
        response_profiles = []
        for profile in user_profiles:
            response_profiles.append(
                UserProfileResponse(
                    username=profile.username,
                    full_name=profile.full_name,  # Added full_name to response
                    age=profile.age,
                    role=profile.role,
                    photo_url=profile.photo_url,
                    message="User profile retrieved successfully"
                )
            )
        
        return response_profiles
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user profiles: {str(e)}")


@router.delete("/profile/{username}", response_model=dict)
async def delete_user_profile(username: str):
    """Delete a user profile by username"""
    try:
        success = user_service.delete_profile(username)
        if not success:
            raise HTTPException(status_code=404, detail=f"User {username} not found")
        
        return {
            "username": username,
            "message": f"User profile {username} deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user profile: {str(e)}")


@router.get("/roles", response_model=List[str])
async def get_all_roles():
    """
    Get all unique user roles in the system
    
    Returns:
        List[str]: A list of all unique role types
    """
    try:
        users_file = os.path.join(settings.DATA_DIR, "users.json")
        
        if not os.path.exists(users_file):
            return []
            
        # Read users data
        with open(users_file, "r") as f:
            users_data = json.load(f)
        
        # Extract all unique roles
        roles = set()
        for username, user_data in users_data.items():
            if "role" in user_data and user_data["role"]:
                roles.add(user_data["role"])
        
        return sorted(list(roles))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving roles: {str(e)}")