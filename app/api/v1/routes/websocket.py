"""
AIDAAN WebSocket Routes.
Enables real-time visual state machine pulses and unified conversational flow.
"""
import asyncio
import json
import logging
import time

from app.services.aidaan.agents.coordinator.coordinator_agent import coordinator_agent
from app.services.persistence import persistence_service
from app.db.repositories import ANONYMOUS_IDENTITIES
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

    greeted = False
    dormant_sent = False
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
            except asyncio.TimeoutError:
                if not dormant_sent:
                    await websocket.send_json({"type": "state", "state": "Dormant"})
                    dormant_sent = True
                continue

            message = json.loads(data)

            # Intent Processing (Unified with REST)
            user_text = message.get("text", "")
            msg_type = message.get("type", "chat")
            conv_id = message.get("conversation_id")
            user_id = message.get(
                "user_id",
                "anonymous-trader"
            )
            context = dict(message.get("context", {}) or {})
            if str(user_id).strip().lower() not in ANONYMOUS_IDENTITIES:
                context.setdefault("username", user_id)

            # Proximity signals drive the Awakening/Dormant states client-side.
            if msg_type == "proximity":
                distance_px = message.get("distance_px")
                try:
                    distance_val = float(distance_px) if distance_px is not None else None
                except Exception:
                    distance_val = None
                if distance_val is not None and distance_val <= 64.0:
                    await websocket.send_json({"type": "state", "state": "Awakening", "detail": "Trader proximity detected."})
                    dormant_sent = False
                else:
                    await websocket.send_json({"type": "state", "state": "Dormant"})
                    dormant_sent = True
                continue

            # If it's an 'init' message, we force a greeting only on the FIRST instance
            if msg_type == "init" and not user_text and not greeted:
                user_text = "Hello AIDAAN"
                greeted = True
            elif msg_type == "init" and not user_text and greeted:
                # Ignore background 'init' pulses once session is active
                continue

            if user_text:
                # Transition to 'Thinking' state only if we have a real payload
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
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"State signal error: {e}")
                    break

                try:
                    async def tool_pulse(tool_name: str):
                        try:
                            await websocket.send_json({
                                "type": "state",
                                "state": "Thinking",
                                "detail": f"Executing {tool_name}..."
                            })
                        except Exception:
                            pass # Connection might have closed

                    response = await coordinator_agent.handle_message(
                        text=user_text,
                        conversation_id=conv_id,
                        context=context,
                        tool_callback=tool_pulse
                    )
                    persistence_service.persist_message_exchange(
                        conversation_id=response.conversation_id,
                        username=None if str(user_id).strip().lower() in ANONYMOUS_IDENTITIES else user_id,
                        user_text=user_text,
                        response_payload=response.model_dump(),
                        channel="websocket",
                        context=context,
                    )

                    # Optional alert signal for risk breaches / urgent desk events
                    lowered = (response.reply or "").lower()
                    if response.model.agent == "risk" and any(k in lowered for k in ["breach", "limit", "urgent", "kill switch"]):
                        await websocket.send_json({"type": "state", "state": "Alert/Busy", "detail": "Risk alert condition detected."})

                    # Stream Final Result
                    await websocket.send_json({
                        "type": "chat_reply",
                        "payload": response.model_dump()
                    })
                    dormant_sent = False
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

            # Do not force Dormant immediately; idle timeout will handle it.

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"WebSocket error: {str(e)}"
        })
