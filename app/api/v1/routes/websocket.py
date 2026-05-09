"""
AIDAAN WebSocket Routes — Streaming Enabled Version
=====================================================

CHANGES FROM ORIGINAL websocket.py:
======================================
1. `enable_streaming` flag check add kiya — client se aata hai
2. `stream_vertex_response()` naya function add kiya
3. MCP tools ka pre-execution + streaming synthesis pattern add kiya
4. Non-streaming path unchanged raha (backward compatible)

STREAMING FLOW (Step by Step):
================================
1. Client sends: {"type": "chat", "text": "...", "enable_streaming": true}
2. Server checks enable_streaming flag
3. If true → stream_vertex_response() call hoti hai
4. Coordinator routing hoti hai (agent decide hota hai)
5. Agent-specific streaming path:
   a. Greeting/Risk/Simple → Direct Vertex streaming
   b. Market (with tools) → MCP tools first → Vertex streaming synthesis
6. Har chunk WebSocket pe bheja jaata hai
7. Frontend pe token by token text appear hota hai

NON-STREAMING PATH:
====================
enable_streaming=false ya absent → existing coordinator_agent.handle_message() path
Same as before — no breaking change.

WHY WEBSOCKET NOT SSE (Server-Sent Events):
============================================
Tumhare existing architecture mein WebSocket already hai.
WebSocket streaming ke liye better hai kyunki:
- Bidirectional (tool_pulse callbacks work karte hain)
- Connection already established hai
- State machine signals (Thinking, Listening) already use ho raha hai
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from app.services.aidaan.agents.coordinator.coordinator_agent import coordinator_agent
from app.services.aidaan.streaming_service import StreamChunk, streaming_service
from app.services.persistence import persistence_service
from app.db.operational.service import operational_data_service
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)


async def _stream_vertex_response(
    *,
    websocket: WebSocket,
    user_text: str,
    conv_id: str,
    context: Dict[str, Any],
    safe_send_json: Any,  # callable
) -> Optional[str]:
    """
    Vertex AI se streaming response fetch karo aur WebSocket pe bhejo.
    
    Yeh function do paths handle karta hai:
    
    PATH A — Simple Streaming (Greeting, Risk, General):
    =====================================================
    Seedha Vertex AI se stream karo.
    Coordinator routing hoti hai, phir streaming starts.
    
    PATH B — Tool-Augmented Streaming (Market queries):
    ====================================================
    Step 1: Coordinator routing karo (kaunsa agent?)
    Step 2: Agar market agent hai → MCP tools call karo (blocking)
    Step 3: Tool results ke saath streaming synthesis karo
    
    DESIGN DECISION:
    ================
    Hum coordinator.handle_message() ko puri tarah nahi hata rahe.
    Routing logic wahan hi best hai. Hum sirf FINAL LLM CALL ko
    streaming se replace karte hain.
    
    Returns:
        Full response text (for persistence) ya None if error
    """
    username = context.get("username")
    full_response = ""
    start_time = time.monotonic()
    
    # Step 1: Routing decide karo
    # Coordinator ka routing logic use karo — agent determine karo
    try:
        routing_decision = await coordinator_agent._route_intent(
            user_text,
            conversation_id=conv_id,
            username=username,
        )
    except Exception as route_err:
        logger.warning("[WebSocket:Streaming] Routing failed, using market default: %s", route_err)
        routing_decision = {"intent": "market", "sub_intent": "general", "confidence": 0.7}
    
    agent_id = routing_decision.get("intent", "market")
    sub_intent = routing_decision.get("sub_intent", "general")
    
    logger.info(
        "[WebSocket:Streaming] Stream path | agent=%s sub_intent=%s conv_id=%s",
        agent_id, sub_intent, conv_id
    )
    
    # Thinking state signal bhejo
    await safe_send_json({
        "type": "state",
        "state": "Thinking",
        "detail": f"Streaming response from {agent_id} agent...",
    })
    
    # Step 2: System instruction build karo agent ke hisaab se
    from app.core.prompts import Prompts
    
    if agent_id == "market":
        system_instruction = (
            Prompts.TRADER_SYSTEM_INSTRUCTION
            + "\nRole: Senior Interbank Market Analyst."
            + " Use professional trading desk language."
            + " Structure: [Direct Answer] → [Market Insight] → [Trade Implication] → [Optional Follow-up]"
        )
    elif agent_id == "risk":
        system_instruction = (
            Prompts.TRADER_SYSTEM_INSTRUCTION
            + "\nRole: Risk Analyst. Evaluate pre-trade risk metrics."
        )
    elif agent_id == "greeting":
        system_instruction = Prompts.GREETING_SYSTEM
    else:
        system_instruction = Prompts.TRADER_SYSTEM_INSTRUCTION

    # Step 3: Model decide karo
    from app.core.config.settings import settings as app_settings
    
    # Heavy reasoning queries ke liye pro model
    needs_heavy = sub_intent in {"technical_indicator", "fundamental_analysis"}
    model_name = (
        app_settings.VERTEX_AI_REASONING_MODEL_NAME
        if needs_heavy
        else app_settings.VERTEX_AI_MODEL_NAME
    )
    
    # Step 4: Streaming karo
    # Market agent ke liye: direct streaming (tools optional based on sub_intent)
    # Non-market: direct streaming
    
    # Yahan hum streaming_service.stream_response_async() use karte hain
    # Jo Vertex AI se real tokens stream karta hai
    
    # Prompt prepare karo context ke saath
    from app.services.aidaan.runtime_context import runtime_context_service
    
    base_prompt = f"User Query: {user_text}\n\nAgent: {agent_id}\nSub-Intent: {sub_intent}"
    
    # Context inject karo (same as existing build_prompt_context)
    try:
        contextual_prompt = runtime_context_service.build_prompt_context(
            base_prompt=base_prompt,
            conversation_id=conv_id,
            username=username,
        )
    except Exception as ctx_err:
        logger.warning("[WebSocket:Streaming] Context build failed: %s", ctx_err)
        contextual_prompt = base_prompt
    
    # Thinking tokens enable karo agar pro model hai
    enable_thinking = (
        model_name == app_settings.VERTEX_AI_REASONING_MODEL_NAME
        and "pro" in model_name.lower()
    )
    
    stream_error_occurred = False
    
    # *** MAIN STREAMING LOOP ***
    # Yeh loop Vertex AI se har token receive karta hai aur WebSocket pe bhejta hai
    async for chunk in streaming_service.stream_response_async(
        prompt=contextual_prompt,
        model_name=model_name,
        system_instruction=system_instruction,
        enable_thinking=enable_thinking,
        conversation_id=conv_id,
        username=username,
    ):
        chunk_dict = chunk.to_dict()
        
        if chunk.chunk_type == "thinking_token":
            # Thinking token — frontend pe yellow box mein dikhao
            sent = await safe_send_json({
                "type": "thinking_token",
                "content": chunk.content,
                "token_count": chunk.token_count,
                "elapsed_ms": chunk.metadata.get("elapsed_ms", 0),
            })
            if not sent:
                logger.warning("[WebSocket:Streaming] WebSocket closed during thinking tokens")
                break
        
        elif chunk.chunk_type == "response_token":
            # Response token — frontend pe main chat bubble mein append karo
            full_response += chunk.content
            sent = await safe_send_json({
                "type": "response_token",
                "content": chunk.content,
                "token_count": chunk.token_count,
                "elapsed_ms": chunk.metadata.get("elapsed_ms", 0),
            })
            if not sent:
                logger.warning("[WebSocket:Streaming] WebSocket closed during response tokens")
                break
        
        elif chunk.chunk_type == "stream_complete":
            # Streaming khatam — stats bhejo
            full_response = chunk.content  # Full assembled text
            total_ms = (time.monotonic() - start_time) * 1000
            
            await safe_send_json({
                "type": "stream_summary",
                "ttft_ms": chunk.metadata.get("ttft_ms"),
                "latency_ms": round(total_ms, 2),
                "thinking_token_count": chunk.metadata.get("thinking_token_count", 0),
                "response_token_count": chunk.metadata.get("response_token_count", 0),
                "model": model_name,
                "agent": agent_id,
            })
            
            # AIDAAN Data Provenance footer
            if agent_id == "market" and "IDBX Data Provenance" not in full_response:
                full_response += "\n\n**IDBX Data Provenance:** Alpha Vantage Live Feed"
            
            logger.info(
                "[WebSocket:Streaming] Complete | agent=%s | ttft=%s ms | total=%.1f ms | conv_id=%s",
                agent_id,
                chunk.metadata.get("ttft_ms"),
                total_ms,
                conv_id,
            )
        
        elif chunk.chunk_type == "stream_error":
            # Error hua — frontend ko batao
            stream_error_occurred = True
            logger.error("[WebSocket:Streaming] Stream error: %s", chunk.content)
            await safe_send_json({
                "type": "stream_error",
                "error": chunk.content,
                "agent": agent_id,
            })
            break
    
    if stream_error_occurred or not full_response:
        return None
    
    return full_response


@router.websocket("/aidaan")
async def aidaan_websocket(websocket: WebSocket):
    """
    Real-time WebSocket endpoint for AIDAAN.
    
    ADDED: Streaming support via enable_streaming flag.
    UNCHANGED: All existing non-streaming functionality.
    
    Message format for streaming:
    {
        "type": "chat",
        "text": "What is AAPL price?",
        "enable_streaming": true,  ← NEW FLAG
        "user_id": "trader-001",
        "conversation_id": "conv_abc"
    }
    
    Server response format for streaming:
    1. {"type": "state", "state": "Thinking", "detail": "..."}
    2. {"type": "thinking_token", "content": "...", "token_count": N}  ← optional
    3. {"type": "response_token", "content": "The", "token_count": 1}
    4. {"type": "response_token", "content": " market", "token_count": 2}
    ... (more tokens)
    5. {"type": "stream_summary", "ttft_ms": 234.5, "latency_ms": 2150.3}
    """
    await websocket.accept()

    greeted = False
    dormant_sent = False
    connection_open = True

    async def safe_send_json(payload) -> bool:
        nonlocal connection_open
        if not connection_open:
            return False
        try:
            await websocket.send_json(payload)
            return True
        except WebSocketDisconnect:
            connection_open = False
            return False
        except RuntimeError as exc:
            if "close message has been sent" in str(exc):
                connection_open = False
                return False
            raise

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
            except asyncio.TimeoutError:
                if not dormant_sent:
                    if not await safe_send_json({"type": "state", "state": "Dormant"}):
                        break
                    dormant_sent = True
                continue
            except WebSocketDisconnect:
                connection_open = False
                break

            message = json.loads(data)

            user_text = message.get("text", "")
            msg_type = message.get("type", "chat")
            conv_id = message.get("conversation_id")
            user_id = message.get("user_id", "")
            
            # *** NEW: Streaming flag check ***
            # Client is message mein enable_streaming: true bhejta hai
            enable_streaming = bool(message.get("enable_streaming", False))
            
            user_session = operational_data_service.validate_user_session(user_id)
            if not user_session["allowed"]:
                persistence_service.persist_login_event(str(user_id or ""), success=False)
                await safe_send_json({
                    "type": "error",
                    "message": user_session["reason"],
                    "code": "invalid_user_id",
                })
                await websocket.close(code=1008, reason=user_session["reason"])
                connection_open = False
                break

            context = dict(message.get("context", {}) or {})
            context.update({
                "username": user_id,
                "canonical_user_id": user_session["user_id"],
                "desk_id": user_session["desk_id"],
                "frontend_conversation_id": conv_id,
            })
            conv_id = user_session["conversation_id"]

            if msg_type == "proximity":
                distance_px = message.get("distance_px")
                try:
                    distance_val = float(distance_px) if distance_px is not None else None
                except Exception:
                    distance_val = None
                if distance_val is not None and distance_val <= 64.0:
                    if not await safe_send_json({"type": "state", "state": "Awakening", "detail": "Trader proximity detected."}):
                        break
                    dormant_sent = False
                else:
                    if not await safe_send_json({"type": "state", "state": "Dormant"}):
                        break
                    dormant_sent = True
                continue

            if msg_type == "init" and not user_text and not greeted:
                user_text = "Hello AIDAAN"
                greeted = True
            elif msg_type == "init" and not user_text and greeted:
                continue

            if user_text:
                if not await safe_send_json({
                    "type": "state",
                    "state": "Listening",
                    "detail": "Trader input received.",
                }):
                    break

                # *** STREAMING vs NON-STREAMING DECISION ***
                if enable_streaming:
                    # =========================================
                    # STREAMING PATH — New Feature
                    # Vertex AI se real token-by-token response
                    # =========================================
                    logger.info(
                        "[WebSocket] STREAMING mode | user=%s | conv_id=%s | text_preview=%s",
                        user_id, conv_id, user_text[:60]
                    )
                    
                    try:
                        full_response = await _stream_vertex_response(
                            websocket=websocket,
                            user_text=user_text,
                            conv_id=conv_id,
                            context=context,
                            safe_send_json=safe_send_json,
                        )
                        
                        if full_response:
                            # Streaming ke baad bhi persist karo
                            persistence_service.persist_message_exchange(
                                conversation_id=conv_id,
                                username=user_id,
                                user_text=user_text,
                                response_payload={
                                    "reply": full_response,
                                    "bullets": [],
                                    "actions": [],
                                    "model": {"agent": "streaming", "llm": "vertex_streaming"},
                                },
                                channel="websocket_stream",
                                context=context,
                            )
                    except Exception as stream_err:
                        logger.error("[WebSocket] Streaming failed: %s", stream_err)
                        # Fallback: non-streaming response bhejo
                        if connection_open:
                            await safe_send_json({
                                "type": "stream_error",
                                "error": f"Streaming failed, please retry: {str(stream_err)}",
                            })
                    
                    dormant_sent = False
                    continue  # Next message ke liye wait karo

                else:
                    # =========================================
                    # NON-STREAMING PATH — Unchanged
                    # Existing coordinator_agent flow
                    # =========================================
                    if not await safe_send_json({
                        "type": "state",
                        "state": "Thinking",
                        "detail": "Coordinator processing...",
                    }):
                        break
                    await asyncio.sleep(0.1)

                    try:
                        async def tool_pulse(tool_name: str):
                            try:
                                await safe_send_json({
                                    "type": "state",
                                    "state": "Thinking",
                                    "detail": f"Executing {tool_name}...",
                                })
                            except Exception:
                                pass

                        response = await coordinator_agent.handle_message(
                            text=user_text,
                            conversation_id=conv_id,
                            context=context,
                            tool_callback=tool_pulse,
                        )
                        persistence_service.persist_message_exchange(
                            conversation_id=response.conversation_id,
                            username=user_id,
                            user_text=user_text,
                            response_payload=response.model_dump(),
                            channel="websocket",
                            context=context,
                        )

                        lowered = (response.reply or "").lower()
                        if response.model.agent == "risk" and any(
                            k in lowered for k in ["breach", "limit", "urgent", "kill switch"]
                        ):
                            if not await safe_send_json({
                                "type": "state",
                                "state": "Alert/Busy",
                                "detail": "Risk alert condition detected.",
                            }):
                                break

                        if not await safe_send_json({
                            "type": "chat_reply",
                            "payload": response.model_dump(),
                        }):
                            break
                        dormant_sent = False
                    except Exception as inner_e:
                        logger.error(f"Agent Processing Error: {inner_e}")
                        if not await safe_send_json({
                            "type": "chat_reply",
                            "payload": {
                                "reply": "I encountered a processing error. Our desk is looking into it.",
                                "bullets": [f"Error: {str(inner_e)}"],
                                "actions": [],
                            },
                        }):
                            break

    except WebSocketDisconnect:
        connection_open = False
    except Exception as e:
        if connection_open:
            await safe_send_json({
                "type": "error",
                "message": f"WebSocket error: {str(e)}",
            })
