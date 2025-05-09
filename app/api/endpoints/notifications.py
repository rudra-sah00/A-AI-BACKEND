from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.websocket_manager import manager
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/notifications")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info(f"Client {websocket.client} connected to notifications websocket.")
    try:
        while True:
            # Keep the connection alive, frontend sends no messages for now
            # Or, you can implement a ping/pong mechanism if needed
            await websocket.receive_text() 
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client {websocket.client} disconnected from notifications websocket.")
    except Exception as e:
        logger.error(f"Error in notifications websocket: {e}")
        manager.disconnect(websocket)
