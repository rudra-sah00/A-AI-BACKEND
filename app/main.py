import sys
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
from contextlib import asynccontextmanager

# Add the parent directory to sys.path if running directly from app/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.core.config import settings
    from app.api.api import api_router
    from app.ai_engine import ai_engine
except ModuleNotFoundError:
    # Fallback to relative imports if running from within the app directory
    from core.config import settings
    from api.api import api_router
    from ai_engine import ai_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend.log')
    ]
)
logger = logging.getLogger("backend")

# Define lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create necessary directories
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    logger.info(f"Application startup: Created necessary directories")
    
    # Start AI Engine
    logger.info("Starting AI Engine...")
    ai_engine.start()
    
    yield  # This is where FastAPI runs and serves requests
    
    # Shutdown: cleanup tasks
    logger.info("Stopping AI Engine...")
    ai_engine.stop()
    
    logger.info("Application shutdown: Cleaning up resources")

# Create the FastAPI app with the lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Camera Management API",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount static files for data directory
print(f"Mounting data files from {settings.DATA_DIR}")
app.mount("/data", StaticFiles(directory=settings.DATA_DIR), name="data")

# Mount static UI files if directory exists
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
if os.path.isdir(static_dir):
    print(f"Mounting static files from {static_dir}")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"Static directory {static_dir} does not exist, skipping static file mounting")

if __name__ == "__main__":
    # Use relative path if running from app directory, absolute path otherwise
    module_path = "main:app" if os.path.basename(os.getcwd()) == "app" else "app.main:app"
    uvicorn.run(module_path, host="0.0.0.0", port=8000, reload=True)