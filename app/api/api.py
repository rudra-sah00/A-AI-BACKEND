from fastapi import APIRouter

from app.api.endpoints import users, cameras, rules, ollama_vision

api_router = APIRouter()
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(cameras.router, prefix="/cameras", tags=["cameras"])
api_router.include_router(rules.router, prefix="/rules", tags=["rules"])
api_router.include_router(ollama_vision.router, prefix="/ollama-vision", tags=["ollama-vision"])