"""
AIDAAN WebSocket Routes — Streaming Enabled Version
=====================================================

STREAMING FLOW (Step by Step):
================================
1. Client sends: {"type": "chat", "text": "...", "enable_streaming": true}
2. Server checks enable_streaming flag
3. If true -> _stream_vertex_response() is called
4. Coordinator routing determines which agent handles the query
5. Agent-specific streaming path:
   a. Market (with tools) -> MCP tools execute first (blocking) -> Vertex streaming synthesis
   b. Greeting / Risk / General -> Direct Vertex streaming (no tools)
6. Each token chunk is sent over WebSocket as it arrives
7. Frontend renders tokens one by one

NON-STREAMING PATH:
====================
enable_streaming=false or absent -> existing coordinator_agent.handle_message() path
No breaking changes to the existing flow.

WHY WEBSOCKET NOT SSE (Server-Sent Events):
============================================
The existing architecture already uses WebSocket for bidirectional communication
(tool_pulse callbacks, state machine signals). Streaming over the same connection
avoids adding a second transport layer.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from app.services.aidaan.agents.coordinator.coordinator_agent import coordinator_agent
from app.services.aidaan.streaming_service import StreamChunk, streaming_service
from app.services.persistence import persistence_service
from app.db.operational.service import operational_data_service
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)


async def _execute_market_tools(
    user_text: str,
    conv_id: str,
    username: Optional[str],
    sub_intent: str,
    safe_send_json: Any,
) -> List[Dict[str, Any]]:
    """
    Execute MCP tools for a market query and return their results.

    This is the blocking phase of Type 2 (Post-Tool) streaming.
    Alpha Vantage API calls happen here synchronously before streaming begins.

    The model decides which tools to call based on the user query.
    We use the existing llm_client MCP session to execute them.

    Args:
        user_text: The user's market query
        conv_id: Conversation ID for context
        username: For DLP and audit
        sub_intent: Routing sub-intent (e.g., "news", "technical_indicator")
        safe_send_json: WebSocket send helper for tool pulse signals

    Returns:
        List of {"tool_name": str, "result": dict} dicts
    """
    from app.core.llm_client import llm_client
    from app.core.config.settings import settings
    from app.core.prompts import Prompts

    tool_results: List[Dict[str, Any]] = []

    # Build a tool-selection prompt — ask the model which tools to call.
    # Include conversation history so the model can resolve tickers from prior context
    # (e.g. user says "last 10 days" after already discussing AAPL).
    from app.services.aidaan.runtime_context import runtime_context_service
    base_tool_prompt = (
        f"User query: {user_text}\n\n"
        f"Sub-intent: {sub_intent}\n\n"
        "You are a market data orchestrator. Based on the user query above, "
        "call the appropriate Alpha Vantage tools to gather the required data. "
        "If the user query is ambiguous or does not mention a ticker, resolve it "
        "from the conversation history provided below. "
        "Execute all necessary tools now. Do not synthesize a final answer yet — "
        "just gather the data."
    )
    try:
        tool_selection_prompt = runtime_context_service.build_prompt_context(
            base_prompt=base_tool_prompt,
            conversation_id=conv_id,
            username=username,
        )
    except Exception:
        tool_selection_prompt = base_tool_prompt

    # Collect tool names as they execute for UI pulse signals
    executed_tools: List[str] = []

    async def tool_pulse(tool_name: str):
        executed_tools.append(tool_name)
        await safe_send_json({
            "type": "state",
            "state": "Thinking",
            "detail": f"Fetching {tool_name}...",
        })

    try:
        # Use the existing MCP tool loop — this is the blocking phase
        # The model calls Alpha Vantage tools, results come back as JSON
        raw_result = await llm_client.generate_json(
            prompt=tool_selection_prompt,
            model_override=settings.VERTEX_AI_MODEL_NAME,
            use_mcp_tools=True,
            tool_callback=tool_pulse,
            system_instruction=(
                Prompts.TRADER_SYSTEM_INSTRUCTION
                + "\nRole: Market data gatherer. Call the required tools and return the raw data."
                + " Do NOT synthesize or explain — just return the tool data as JSON."
            ),
        )

        # Package the result for stream_after_tools
        tool_results.append({
            "tool_name": "market_data_bundle",
            "result": raw_result,
        })

        logger.info(
            "[WebSocket:Streaming] Market tools completed | tools_called=%s | conv_id=%s",
            executed_tools,
            conv_id,
        )

    except Exception as tool_err:
        logger.warning("[WebSocket:Streaming] Tool execution failed: %s", tool_err)
        # Return empty list — streaming will proceed without tool data
        # The synthesis model will acknowledge the missing data

    return tool_results


async def _stream_vertex_response(
    *,
    websocket: WebSocket,
    user_text: str,
    conv_id: str,
    context: Dict[str, Any],
    safe_send_json: Any,
) -> Optional[str]:
    """
    Fetch a streaming response from Vertex AI and send it over WebSocket.

    Handles two paths:

    PATH A — Tool-Augmented Streaming (Market queries):
    ---------------------------------------------------
    Step 1: Coordinator routing (which agent?)
    Step 2: MCP tools execute (Alpha Vantage calls, blocking ~1-2s)
    Step 3: stream_after_tools() streams the synthesis using tool results
    User sees tokens immediately after tools complete.

    PATH B — Direct Streaming (Greeting, Risk, General):
    ----------------------------------------------------
    Step 1: Coordinator routing
    Step 2: stream_response_async() streams directly from Vertex AI
    No tool phase.

    Args:
        websocket: The active WebSocket connection
        user_text: The user's message
        conv_id: Conversation ID
        context: Session context dict (username, desk_id, etc.)
        safe_send_json: WebSocket send helper that handles disconnects gracefully

    Returns:
        Full assembled response text (for persistence), or None on error
    """
    username = context.get("username")
    full_response = ""
    start_time = time.monotonic()

    # Step 1: Determine which agent handles this query
    try:
        routing_decision = await coordinator_agent._route_intent(
            user_text,
            conversation_id=conv_id,
            username=username,
        )
    except Exception as route_err:
        logger.warning("[WebSocket:Streaming] Routing failed, defaulting to market: %s", route_err)
        routing_decision = {"intent": "market", "sub_intent": "general", "confidence": 0.7}

    agent_id = routing_decision.get("intent", "market")
    sub_intent = routing_decision.get("sub_intent", "general")

    logger.info(
        "[WebSocket:Streaming] Stream path | agent=%s sub_intent=%s conv_id=%s",
        agent_id, sub_intent, conv_id
    )

    # Signal to frontend that processing has started
    await safe_send_json({
        "type": "state",
        "state": "Thinking",
        "detail": f"Routing to {agent_id} agent...",
    })

    # Step 2: Build system instruction for the agent
    from app.core.prompts import Prompts

    if agent_id == "market":
        system_instruction = (
            Prompts.TRADER_SYSTEM_INSTRUCTION
            + "\nRole: Senior Interbank Market Analyst."
            + " Use professional trading desk language."
            + " Structure: [Direct Answer] -> [Market Insight] -> [Trade Implication] -> [Optional Follow-up]"
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

    # Step 3: Select model
    from app.core.config.settings import settings as app_settings

    needs_heavy = sub_intent in {"technical_indicator", "fundamental_analysis"}
    model_name = (
        app_settings.VERTEX_AI_REASONING_MODEL_NAME
        if needs_heavy
        else app_settings.VERTEX_AI_MODEL_NAME
    )

    # Step 4: Build contextual prompt
    from app.services.aidaan.runtime_context import runtime_context_service

    base_prompt = f"User Query: {user_text}\n\nAgent: {agent_id}\nSub-Intent: {sub_intent}"

    try:
        contextual_prompt = runtime_context_service.build_prompt_context(
            base_prompt=base_prompt,
            conversation_id=conv_id,
            username=username,
        )
    except Exception as ctx_err:
        logger.warning("[WebSocket:Streaming] Context build failed: %s", ctx_err)
        contextual_prompt = base_prompt

    # Thinking tokens only for pro reasoning models
    enable_thinking = (
        model_name == app_settings.VERTEX_AI_REASONING_MODEL_NAME
        and "pro" in model_name.lower()
    )

    stream_error_occurred = False

    # =========================================================================
    # PATH A: Market agent — tools first, then stream synthesis
    # =========================================================================
    if agent_id == "market":
        await safe_send_json({
            "type": "state",
            "state": "Thinking",
            "detail": "Fetching market data...",
        })

        # Execute MCP tools (blocking phase — Alpha Vantage calls happen here)
        tool_results = await _execute_market_tools(
            user_text=user_text,
            conv_id=conv_id,
            username=username,
            sub_intent=sub_intent,
            safe_send_json=safe_send_json,
        )

        await safe_send_json({
            "type": "state",
            "state": "Thinking",
            "detail": "Streaming synthesis...",
        })

        # Stream the synthesis using tool results
        # User sees tokens immediately after tools complete
        async for chunk in streaming_service.stream_after_tools(
            tool_results=tool_results,
            original_prompt=contextual_prompt,
            system_instruction=system_instruction,
            model_name=model_name,
            conversation_id=conv_id,
            username=username,
        ):
            if chunk.chunk_type == "thinking_token":
                sent = await safe_send_json({
                    "type": "thinking_token",
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "elapsed_ms": chunk.metadata.get("elapsed_ms", 0),
                })
                if not sent:
                    break

            elif chunk.chunk_type == "response_token":
                full_response += chunk.content
                sent = await safe_send_json({
                    "type": "response_token",
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "elapsed_ms": chunk.metadata.get("elapsed_ms", 0),
                })
                if not sent:
                    break

            elif chunk.chunk_type == "stream_complete":
                full_response = chunk.content
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

                if "IDBX Data Provenance" not in full_response:
                    full_response += "\n\n**IDBX Data Provenance:** Alpha Vantage Live Feed"

                logger.info(
                    "[WebSocket:Streaming] Market stream complete | ttft=%s ms | total=%.1f ms | conv_id=%s",
                    chunk.metadata.get("ttft_ms"),
                    total_ms,
                    conv_id,
                )

            elif chunk.chunk_type == "stream_error":
                stream_error_occurred = True
                logger.error("[WebSocket:Streaming] Market stream error: %s", chunk.content)
                await safe_send_json({
                    "type": "stream_error",
                    "error": chunk.content,
                    "agent": agent_id,
                })
                break

    # =========================================================================
    # PATH B: Non-market agents — direct streaming, no tools
    # =========================================================================
    else:
        async for chunk in streaming_service.stream_response_async(
            prompt=contextual_prompt,
            model_name=model_name,
            system_instruction=system_instruction,
            enable_thinking=enable_thinking,
            conversation_id=conv_id,
            username=username,
        ):
            if chunk.chunk_type == "thinking_token":
                sent = await safe_send_json({
                    "type": "thinking_token",
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "elapsed_ms": chunk.metadata.get("elapsed_ms", 0),
                })
                if not sent:
                    break

            elif chunk.chunk_type == "response_token":
                full_response += chunk.content
                sent = await safe_send_json({
                    "type": "response_token",
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "elapsed_ms": chunk.metadata.get("elapsed_ms", 0),
                })
                if not sent:
                    break

            elif chunk.chunk_type == "stream_complete":
                full_response = chunk.content
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

                logger.info(
                    "[WebSocket:Streaming] Direct stream complete | agent=%s | ttft=%s ms | total=%.1f ms | conv_id=%s",
                    agent_id,
                    chunk.metadata.get("ttft_ms"),
                    total_ms,
                    conv_id,
                )

            elif chunk.chunk_type == "stream_error":
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

    Streaming is opt-in via the enable_streaming flag in the message payload.
    All existing non-streaming functionality is unchanged.

    Message format for streaming:
    {
        "type": "chat",
        "text": "What is AAPL price?",
        "enable_streaming": true,
        "user_id": "trader-001",
        "conversation_id": "conv_abc"
    }

    Server response sequence for streaming:
    1. {"type": "state", "state": "Thinking", "detail": "..."}
    2. {"type": "state", "state": "Thinking", "detail": "Fetching get_stock_quote..."}  <- tool pulse
    3. {"type": "response_token", "content": "AAPL", "token_count": 1}
    4. {"type": "response_token", "content": " is", "token_count": 2}
    ... (more tokens)
    5. {"type": "stream_summary", "ttft_ms": 234.5, "latency_ms": 2150.3, ...}
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

            # Streaming opt-in flag — client sends enable_streaming: true to use streaming path
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

                if enable_streaming:
                    # =========================================================
                    # STREAMING PATH
                    # Type 2: Tools execute first (blocking), then Vertex streams
                    # =========================================================
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
                        if connection_open:
                            await safe_send_json({
                                "type": "stream_error",
                                "error": f"Streaming failed, please retry: {str(stream_err)}",
                            })

                    dormant_sent = False
                    continue

                else:
                    # =========================================================
                    # NON-STREAMING PATH — Unchanged
                    # Existing coordinator_agent flow
                    # =========================================================
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
