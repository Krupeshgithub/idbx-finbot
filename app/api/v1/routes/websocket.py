"""
AIDAAN WebSocket Routes.
Enables real-time visual state machine pulses and low-latency interaction.
"""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect


router = APIRouter()


@router.websocket("/aidaan")
async def aidaan_websocket(websocket: WebSocket):
    """
    Real-time WebSocket endpoint for AIDAAN.
    Signals sphere states (Listening, Thinking) and streams responses.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 1. Pulse 'Listening'
            await websocket.send_json({"state": "Listening", "detail": "Trader input received."})
            
            # 2. Simulate 'Thinking'
            await websocket.send_json({"state": "Thinking", "detail": "Processing intent..."})
            await asyncio.sleep(1)
            
            # 3. Simulate Agent Response
            await websocket.send_json({
                "state": "Awakening",
                "reply": "I'm processing that for you. Use the REST API for formal conversational turns in Phase 1.",
                "bullets": ["Visual State Machine: Active", "Institutional Connectivity: Normal"]
            })
            
            # 4. Return to 'Dormant' (implicit)
            await websocket.send_json({"state": "Dormant"})
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})
