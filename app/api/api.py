from fastapi import APIRouter

from app.api.endpoints import ai_vision, users, cameras, rules, notifications

api_router = APIRouter()
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(cameras.router, prefix="/cameras", tags=["cameras"])
api_router.include_router(rules.router, prefix="/rules", tags=["rules"])
api_router.include_router(ai_vision.router, prefix="/contextual", tags=["contextual"])
api_router.include_router(notifications.router, prefix="/notifications", tags=["notifications"])