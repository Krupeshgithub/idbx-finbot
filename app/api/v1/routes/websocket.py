"""
AIDAAN WebSocket Routes.
Enables real-time visual state machine pulses and unified conversational flow.
"""
import asyncio
import json

from app.services.aidaan.coordinator import coordinator_agent
from app.schemas.aidaan import AidaanMessageRequest
from fastapi import (
    APIRouter, 
    WebSocket, 
    WebSocketDisconnect
)


router = APIRouter()


@router.websocket("/aidaan")
async def aidaan_websocket(websocket: WebSocket):
    """
    Real-time WebSocket endpoint for AIDAAN.
    Signals sphere states (Listening, Thinking) and handles unified chat.
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # VisualPulse: Listening
            await websocket.send_json({
                "type": "state", 
                "state": "Listening", 
                "detail": "Trader input received."
            })

            # VisualPulse: Thinking
            await websocket.send_json({
                "type": "state",
                "state": "Thinking",
                "detail": "Coordinator processing..."
            })
            await asyncio.sleep(0.5)

            # Intent Processing (Unified with REST)
            user_text = message.get("text", "")
            conv_id = message.get("conversation_id")
            user_id = message.get(
                "user_id",
                "anonymous-trader"
            )

            if user_text:
                response = await coordinator_agent.handle_message(
                    text=user_text,
                    conversation_id=conv_id,
                    context=message.get("context", {})
                )

                # Stream Final Result
                # We return the same structured AidaanMessageResponse as the REST API
                await websocket.send_json({
                    "type": "chat_reply",
                    "payload": response.model_dump()
                })

            await websocket.send_json({
                "type": "state",
                "state": "Dormant"
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"WebSocket error: {str(e)}"
        })
