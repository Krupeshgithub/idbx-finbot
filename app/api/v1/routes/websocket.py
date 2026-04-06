"""
AIDAAN WebSocket Routes.
Enables real-time visual state machine pulses and unified conversational flow.
"""
import asyncio
import json
import logging

from app.services.aidaan.coordinator import coordinator_agent
from app.schemas.aidaan import AidaanMessageRequest
from fastapi import (
    APIRouter, 
    WebSocket, 
    WebSocketDisconnect
)


router = APIRouter()
logger = logging.getLogger(__name__)


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

            print("Received WebSocket message:")
            print(message)

            # Safety wrap for state signals
            try:
                await websocket.send_json({
                    "type": "state", 
                    "state": "Listening", 
                    "detail": "Trader input received."
                })

                await websocket.send_json({
                    "type": "state",
                    "state": "Thinking",
                    "detail": "Coordinator processing..."
                })
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"State signal error: {e}")
                break # Exit if we can't even send state pulses

            # Intent Processing (Unified with REST)
            user_text = message.get("text", "")
            conv_id = message.get("conversation_id")
            user_id = message.get(
                "user_id",
                "anonymous-trader"
            )

            if user_text:
                try:
                    response = await coordinator_agent.handle_message(
                        text=user_text,
                        conversation_id=conv_id,
                        context=message.get("context", {})
                    )

                    # Stream Final Result
                    await websocket.send_json({
                        "type": "chat_reply",
                        "payload": response.model_dump()
                    })
                except Exception as inner_e:
                    logger.error(f"Agent Processing Error: {inner_e}")
                    await websocket.send_json({
                        "type": "chat_reply",
                        "payload": {
                            "reply": "I encountered a processing error. Our desk is looking into it.",
                            "bullets": [f"Error: {str(inner_e)}"],
                            "actions": []
                        }
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
