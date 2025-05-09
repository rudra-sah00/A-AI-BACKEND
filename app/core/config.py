from pydantic import Field, computed_field
from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path


class Settings(BaseSettings):
    # Base directory of the project
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI-Backend"
    
    # Base URL for frontend links - read from .env file
    BACKEND_URL: str = Field(default="http://localhost:8000", env="BACKEND_URL")
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    
    # Data storage - use string path that will be converted to Path
    DATA_DIR_STR: str = Field(default="data", env="DATA_DIR")
    
    # AI model settings
    AI_MODEL_PATH: str = Field(default="app/ai/models")
    ENABLE_GPU: bool = Field(default=False)
    AI_ENGINE_MONITOR_INTERVAL_SECONDS: int = Field(default=30, env="AI_ENGINE_MONITOR_INTERVAL_SECONDS")
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]

    class Config:
        env_file = ".env"
        case_sensitive = True
    
    # Use a computed property for DATA_DIR
    @computed_field
    @property
    def DATA_DIR(self) -> Path:
        """Return absolute path to data directory"""
        # If DATA_DIR_STR is absolute, use it directly
        path = Path(self.DATA_DIR_STR)
        if path.is_absolute():
            return path
        # Otherwise, make it relative to BASE_DIR
        return self.BASE_DIR / self.DATA_DIR_STR


settings = Settings()

# Ensure data directory exists
os.makedirs(settings.DATA_DIR, exist_ok=True)
print(f"Using data directory: {settings.DATA_DIR}")