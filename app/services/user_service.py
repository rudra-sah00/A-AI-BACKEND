import json
import os
import uuid
import re
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.models.user import UserProfile, UserProfileCreate


class UserService:
    def __init__(self):
        # Ensure the data directory exists
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        self.users_file = settings.DATA_DIR / "users.json"
        
        # Initialize users.json if it doesn't exist
        if not os.path.exists(self.users_file):
            with open(self.users_file, "w") as f:
                json.dump({}, f)
    
    def _read_users(self) -> Dict:
        """Read users from JSON file"""
        try:
            with open(self.users_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is empty/invalid JSON, return empty dict
            return {}
    
    def _write_users(self, users_data: Dict) -> None:
        """Write users to JSON file"""
        with open(self.users_file, "w") as f:
            json.dump(users_data, f, indent=4)
    
    def create_or_update_profile(self, user_data: UserProfileCreate) -> UserProfile:
        """Create or update a user profile"""
        users = self._read_users()
        
        # Convert file path to URL format
        photo_url = None
        relative_path = None
        if user_data.photo_path:
            # Extract the unique ID and username from the path
            # Path format: data/unique_id/username.ext
            path_parts = Path(user_data.photo_path).parts
            if len(path_parts) >= 3:
                # If it's already in the right format
                unique_id = path_parts[-2]  # Second-to-last part (the unique ID folder)
                filename = path_parts[-1]   # Last part (username.ext)
                relative_path = f"data/{unique_id}/{filename}"
            else:
                # Handle absolute paths or other formats
                abs_path = Path(user_data.photo_path).absolute()
                if "data" in abs_path.parts:
                    data_index = abs_path.parts.index("data")
                    if len(abs_path.parts) > data_index + 2:
                        unique_id = abs_path.parts[data_index + 1]
                        filename = abs_path.parts[data_index + 2]
                        relative_path = f"data/{unique_id}/{filename}"
            
            # Create full URL
            if relative_path:
                photo_url = f"{settings.BACKEND_URL}/{relative_path}"
        
        # Check if user already exists
        if user_data.username in users:
            # Update existing user
            user_dict = users[user_data.username]
            user_dict["age"] = user_data.age
            user_dict["role"] = user_data.role
            user_dict["full_name"] = user_data.full_name  # Update full_name
            
            # Only update photo URL if a new one is provided
            if photo_url:
                user_dict["photo_url"] = photo_url
                # Also keep the photo_path for backward compatibility and file operations
                user_dict["photo_path"] = relative_path if relative_path else user_dict.get("photo_path")
            
            user_dict["updated_at"] = datetime.now().isoformat()
        else:
            # Create new user
            users[user_data.username] = {
                "id": str(uuid.uuid4()),
                "username": user_data.username,
                "full_name": user_data.full_name,  # Add full_name
                "age": user_data.age,
                "role": user_data.role,
                "photo_url": photo_url,
                "photo_path": relative_path,
                "created_at": user_data.created_at,
                "updated_at": None
            }
        
        # Save changes
        self._write_users(users)
        
        # Return user profile
        user_dict = users[user_data.username]
        return UserProfile(
            id=user_dict["id"],
            username=user_dict["username"],
            full_name=user_dict["full_name"],  # Include full_name
            age=user_dict["age"],
            role=user_dict["role"],
            photo_path=user_dict.get("photo_path"),
            photo_url=user_dict.get("photo_url"),
            created_at=user_dict["created_at"],
            updated_at=user_dict.get("updated_at")
        )
    
    def get_profile(self, username: str) -> Optional[UserProfile]:
        """Get a user profile by username"""
        users = self._read_users()
        if username not in users:
            return None
        
        return UserProfile(**users[username])
    
    def delete_profile(self, username: str) -> bool:
        """Delete a user profile by username"""
        users = self._read_users()
        if username not in users:
            return False
        
        # Get user data before deleting
        user = users[username]
        
        # Delete the user's photo if it exists
        photo_path = user.get("photo_path")
        if photo_path and photo_path.startswith("data/"):
            # Extract the unique ID directory from the path
            # Format: data/unique_id/username.ext
            parts = photo_path.split("/")
            if len(parts) >= 3:
                unique_id = parts[1]  # The unique ID is the second part
                unique_dir = settings.DATA_DIR / unique_id
                
                if unique_dir.exists() and unique_dir.is_dir():
                    import shutil
                    shutil.rmtree(unique_dir)
        
        # Delete user from users.json
        del users[username]
        self._write_users(users)
        
        return True
    
    def list_profiles(self) -> List[UserProfile]:
        """List all user profiles"""
        users = self._read_users()
        return [UserProfile(**user_data) for user_data in users.values()]