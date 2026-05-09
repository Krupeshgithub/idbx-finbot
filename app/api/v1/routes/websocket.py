"""
AIDAAN WebSocket Routes — Streaming Enabled Version
=====================================================

FULL PIPELINE VISIBILITY (1% to 100%):
========================================
Every step from user input to streaming response is tracked and emitted
as a pipeline_event message so the frontend shows a live process box.

Step breakdown:
   5% — Input received, DLP check
  10% — Coordinator routing started
  20% — Routing complete (agent + sub_intent decided)
  25% — Context + conversation history loaded
  30% — Tool orchestration started (market queries only)
  30-85% — Each tool start/done (distributed across tool count)
  90% — All tools done, synthesis starting
  95% — Vertex AI streaming first token
 100% — Response complete

PIPELINE_EVENT MESSAGE PROTOCOL:
==================================
{"type":"pipeline_event","step":"routing_start",   "pct":10,"label":"Coordinator routing query..."}
{"type":"pipeline_event","step":"routing_done",    "pct":20,"label":"-> market agent  fx_analysis  via heuristic"}
{"type":"pipeline_event","step":"context_loaded",  "pct":25,"label":"Context loaded — 6 history messages"}
{"type":"pipeline_event","step":"tools_start",     "pct":30,"label":"Tool orchestration started"}
{"type":"pipeline_event","step":"tool_start",      "pct":35,"label":"get_exchange_rate  EUR","tool":"get_exchange_rate","arg":"EUR"}
{"type":"pipeline_event","step":"tool_done",       "pct":60,"label":"get_exchange_rate  EUR/USD=1.08  1.1s","elapsed_s":1.1}
{"type":"pipeline_event","step":"synthesis_start", "pct":90,"label":"Gemini synthesising from 1 tool..."}
{"type":"pipeline_event","step":"streaming",       "pct":95,"label":"Streaming response..."}
{"type":"pipeline_event","step":"done",            "pct":100,"label":"Complete"}

NON-STREAMING PATH:
====================
enable_streaming=false -> existing coordinator_agent.handle_message() path unchanged.
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


# ---------------------------------------------------------------------------
# Preview helper
# ---------------------------------------------------------------------------

def _make_tool_preview(result: Any) -> str:
    """Extract a short human-readable preview from a tool result dict."""
    if not isinstance(result, dict):
        return str(result)[:80]
    if "price" in result and "symbol" in result:
        return f"{result['symbol']} ${result.get('price','')} {result.get('change_percent','')}".strip()
    if "data" in result and isinstance(result.get("data"), list):
        return f"{result.get('symbol','')} — {len(result['data'])} rows"
    if "feed" in result:
        return f"{len(result.get('feed',[]))} articles | FinBERT: {result.get('analytics_status','')}"
    if "rate" in result:
        return f"{result.get('from','?')}/{result.get('to','?')} = {result.get('rate','?')}"
    if "error" in result:
        return f"Error: {str(result['error'])[:60]}"
    for k, v in result.items():
        if isinstance(v, (str, int, float)) and v:
            return f"{k}: {str(v)[:60]}"
    return "Data received"


# ---------------------------------------------------------------------------
# Pipeline emitter
# ---------------------------------------------------------------------------

class _PipelineEmitter:
    """
    Emits pipeline_event messages over WebSocket for every meaningful step
    in the request lifecycle, from routing to streaming completion.
    """

    _TOOL_LABELS = {
        "get_stock_quote":              "Fetching live quote",
        "get_daily_series":             "Fetching daily OHLCV",
        "get_weekly_series":            "Fetching weekly OHLCV",
        "get_intraday_series":          "Fetching intraday data",
        "get_monthly_series":           "Fetching monthly OHLCV",
        "get_market_news":              "Fetching news + FinBERT sentiment",
        "get_company_overview":         "Fetching company fundamentals",
        "get_earnings":                 "Fetching earnings data",
        "get_income_statement":         "Fetching income statement",
        "get_balance_sheet":            "Fetching balance sheet",
        "get_cash_flow":                "Fetching cash flow",
        "get_technical_indicator":      "Computing technical indicator",
        "get_sma":                      "Computing SMA",
        "get_ema":                      "Computing EMA",
        "get_rsi":                      "Computing RSI",
        "get_macd":                     "Computing MACD",
        "get_exchange_rate":            "Fetching FX rate",
        "get_fx_daily_series":          "Fetching FX daily series",
        "get_fx_intraday_series":       "Fetching FX intraday",
        "get_crypto_daily_series":      "Fetching crypto data",
        "get_economic_indicator":       "Fetching economic indicator",
        "get_commodity_price":          "Fetching commodity price",
        "search_ticker":                "Searching ticker symbol",
        "get_top_stocks_by_market_cap": "Fetching top stocks by market cap",
        "get_top_gainers":              "Fetching top gainers",
        "get_top_losers":               "Fetching top losers",
        "get_top_stocks_by_volume":     "Fetching most active stocks",
        "consult_specialist_agent":     "Consulting specialist agent",
    }

    def __init__(self, safe_send_json: Any):
        self._send = safe_send_json
        self._tool_starts: Dict[str, float] = {}
        self._completed_tools: List[str] = []
        self._total_tools_expected: int = 0

    async def _emit(self, step: str, pct: int, label: str, **extra) -> None:
        await self._send({
            "type": "pipeline_event",
            "step": step,
            "pct": min(100, max(0, pct)),
            "label": label,
            **extra,
        })

    async def input_received(self) -> None:
        await self._emit("input_received", 5, "Input received — DLP check")

    async def routing_start(self) -> None:
        await self._emit("routing_start", 10, "Coordinator routing query...")

    async def routing_done(self, agent: str, sub_intent: str, method: str) -> None:
        method_tag = "heuristic" if method == "heuristic" else "LLM"
        await self._emit(
            "routing_done", 20,
            f"-> {agent} agent  \u00b7  {sub_intent}  \u00b7  via {method_tag}",
            agent=agent, sub_intent=sub_intent, method=method,
        )

    async def context_loaded(self, history_count: int) -> None:
        label = (
            f"Context loaded \u2014 {history_count} history messages"
            if history_count > 0
            else "Context loaded \u2014 fresh session"
        )
        await self._emit("context_loaded", 25, label, history_count=history_count)

    async def tools_start(self) -> None:
        await self._emit("tools_start", 30, "Tool orchestration started \u2014 calling Alpha Vantage")

    def _tool_pct(self, done: int, total: int, phase: str) -> int:
        if total == 0:
            return 30
        band = 55  # 85 - 30
        base = 30
        if phase == "start":
            return base + round((done / total) * band * 0.4)
        return base + round((done / total) * band)

    async def tool_start(self, tool_name: str, args: Dict[str, Any]) -> None:
        self._tool_starts[tool_name] = time.monotonic()
        self._total_tools_expected = max(
            self._total_tools_expected,
            len(self._tool_starts) + len(self._completed_tools),
        )
        idx = len(self._completed_tools) + len(self._tool_starts)
        total = self._total_tools_expected

        primary = (
            args.get("symbol") or args.get("keywords") or
            args.get("tickers") or args.get("from_currency") or
            args.get("function") or ""
        )
        hint = self._TOOL_LABELS.get(tool_name, tool_name)
        label = f"{tool_name}  {str(primary).upper()}" if primary else tool_name

        await self._emit(
            "tool_start",
            self._tool_pct(idx - 1, total, "start"),
            label,
            tool=tool_name,
            arg=str(primary).upper() if primary else "",
            hint=hint,
            tool_index=idx,
            tool_total=total,
        )

    async def tool_done(self, tool_name: str, result: Any) -> None:
        start = self._tool_starts.pop(tool_name, time.monotonic())
        elapsed = round(time.monotonic() - start, 2)
        self._completed_tools.append(tool_name)
        done = len(self._completed_tools)
        total = max(self._total_tools_expected, done)
        preview = _make_tool_preview(result) if isinstance(result, dict) else str(result)[:80]
        pct = self._tool_pct(done, total, "done")
        progress_pct = round((done / total) * 100) if total else 100

        await self._emit(
            "tool_done", pct,
            f"{tool_name}  \u00b7  {preview}  \u00b7  {elapsed}s",
            tool=tool_name,
            preview=preview,
            elapsed_s=elapsed,
            tool_index=done,
            tool_total=total,
            progress_pct=progress_pct,
        )

    async def synthesis_start(self) -> None:
        n = len(self._completed_tools)
        await self._emit(
            "synthesis_start", 90,
            f"Gemini synthesising from {n} tool{'s' if n != 1 else ''}...",
            tools_completed=n,
            tools_list=self._completed_tools,
        )

    async def streaming_started(self) -> None:
        await self._emit("streaming", 95, "Streaming response...")

    async def done(self) -> None:
        await self._emit("done", 100, "Complete")

    async def __call__(self, tool_name: str) -> None:
        """Called by llm_client with just the tool name (existing interface)."""
        await self.tool_start(tool_name, {})


# ---------------------------------------------------------------------------
# Market tool execution
# ---------------------------------------------------------------------------

async def _execute_market_tools(
    user_text: str,
    conv_id: str,
    username: Optional[str],
    sub_intent: str,
    emitter: _PipelineEmitter,
) -> List[Dict[str, Any]]:
    """
    Execute MCP tools for a market query and return their results.
    Emits pipeline_event messages for each tool start/done via the emitter.
    """
    from app.core.llm_client import llm_client
    from app.core.config.settings import settings
    from app.core.prompts import Prompts
    from app.services.aidaan.runtime_context import runtime_context_service

    tool_results: List[Dict[str, Any]] = []

    base_tool_prompt = (
        f"User query: {user_text}\n\nSub-intent: {sub_intent}\n\n"
        "You are a market data orchestrator. Call the appropriate Alpha Vantage tools "
        "to gather the required data. Resolve tickers from conversation history if needed. "
        "Execute all necessary tools now. Do not synthesize — just gather data."
    )
    try:
        tool_selection_prompt = runtime_context_service.build_prompt_context(
            base_prompt=base_tool_prompt,
            conversation_id=conv_id,
            username=username,
        )
    except Exception:
        tool_selection_prompt = base_tool_prompt

    try:
        raw_result = await llm_client.generate_json(
            prompt=tool_selection_prompt,
            model_override=settings.VERTEX_AI_MODEL_NAME,
            use_mcp_tools=True,
            tool_callback=emitter,
            system_instruction=(
                Prompts.TRADER_SYSTEM_INSTRUCTION
                + "\nRole: Market data gatherer. Call tools and return raw data as JSON."
                + "\nTICKER RESOLUTION: Apple->AAPL, Google->GOOGL, Microsoft->MSFT, "
                + "Amazon->AMZN, Tesla->TSLA, Meta->META, Nvidia->NVDA, Exxon->XOM, "
                + "Netflix->NFLX, Uber->UBER. Only call search_ticker for unknown companies."
                + "\nPARALLEL: Call all required tools in one batch, not sequentially."
                + "\nOUTPUT: Return a single JSON object with all results."
            ),
        )
        tool_results.append({"tool_name": "market_data_bundle", "result": raw_result})
        logger.info("[WebSocket:Streaming] Tools done | tools=%s | conv_id=%s",
                    emitter._completed_tools, conv_id)
    except Exception as tool_err:
        logger.warning("[WebSocket:Streaming] Tool execution failed: %s", tool_err)

    return tool_results


# ---------------------------------------------------------------------------
# Main streaming handler
# ---------------------------------------------------------------------------

async def _stream_vertex_response(
    *,
    websocket: WebSocket,
    user_text: str,
    conv_id: str,
    context: Dict[str, Any],
    safe_send_json: Any,
) -> Optional[str]:
    """
    Full streaming pipeline with live progress events from 5% to 100%.

    PATH A — Market (tools + synthesis streaming):
      input_received -> routing -> context_loaded -> tools_start ->
      tool_start/done x N -> synthesis_start -> streaming -> done

    PATH B — Non-market (direct streaming):
      input_received -> routing -> context_loaded -> streaming -> done
    """
    username = context.get("username")
    full_response = ""
    start_time = time.monotonic()
    emitter = _PipelineEmitter(safe_send_json)

    # Step 1 — input received
    await emitter.input_received()

    # Step 2 — routing
    await emitter.routing_start()
    routing_method = "llm"
    try:
        routing_decision = await coordinator_agent._route_intent(
            user_text,
            conversation_id=conv_id,
            username=username,
        )
        if routing_decision.get("confidence", 0) >= 0.95:
            routing_method = "heuristic"
    except Exception as route_err:
        logger.warning("[WebSocket:Streaming] Routing failed, defaulting to market: %s", route_err)
        routing_decision = {"intent": "market", "sub_intent": "general", "confidence": 0.7}

    agent_id = routing_decision.get("intent", "market")
    sub_intent = routing_decision.get("sub_intent", "general")

    await emitter.routing_done(agent_id, sub_intent, routing_method)
    logger.info("[WebSocket:Streaming] Stream path | agent=%s sub_intent=%s conv_id=%s",
                agent_id, sub_intent, conv_id)

    # Step 3 — context history count (for the pipeline event only)
    from app.services.aidaan.runtime_context import runtime_context_service
    history_count = 0
    try:
        history = runtime_context_service.get_recent_history(conv_id)
        history_count = len(history) if history else 0
    except Exception:
        pass
    await emitter.context_loaded(history_count)

    stream_error_occurred = False

    # =========================================================================
    # PATH A: Market — run market agent with REAL token-by-token streaming
    #
    # Phase 1 (Blocking): MCP tools gather complete data
    # Phase 2 (Streaming): Synthesis streams token-by-token from Vertex AI
    #
    # This gives us:
    # - Identical tool selection and data gathering as non-streaming
    # - Real TTFT (~500ms-1s instead of 2-3s)
    # - True token-by-token streaming (not simulated word batches)
    # =========================================================================
    if agent_id == "market":
        await emitter.tools_start()

        # Run market agent with streaming enabled
        from app.services.aidaan.agents.market.market_agent import market_agent as _market_agent

        agent_response = None
        streaming_chunks = None
        agent_error: Optional[str] = None
        try:
            result = await _market_agent.handle_message(
                text=user_text,
                conversation_id=conv_id,
                context={
                    **context,
                    "routing": routing_decision,
                },
                tool_callback=emitter,
                enable_streaming=True,  # Enable real token-by-token streaming
            )
            # Unpack tuple: (response_metadata, streaming_generator)
            if isinstance(result, tuple) and len(result) == 2:
                agent_response, streaming_chunks = result
                logger.info(f"[WebSocket:Streaming] ✅ Tuple unpacked successfully | has_streaming_chunks={streaming_chunks is not None}")
            else:
                # Fallback: non-streaming response
                agent_response = result
                streaming_chunks = None
                logger.warning(f"[WebSocket:Streaming] ⚠️ Result is NOT a tuple | type={type(result)} | falling back to non-streaming")
        except Exception as agent_exc:
            logger.error("[WebSocket:Streaming] Market agent failed: %s", agent_exc)
            agent_error = str(agent_exc)

        await emitter.synthesis_start()

        # If the agent failed, surface a clean error chunk
        if agent_error or agent_response is None:
            await safe_send_json({
                "type": "stream_error",
                "error": agent_error or "Market agent returned no response",
                "agent": agent_id,
            })
            return None

        # Check if streaming_chunks generator is available
        if streaming_chunks is None:
            logger.warning(f"[WebSocket:Streaming] ⚠️ streaming_chunks is None - using fallback word-by-word streaming")
            # Fallback: agent returned a pre-built response (shouldn't happen with enable_streaming=True)
            reply_text = (agent_response.reply or "").strip()
            if not reply_text:
                reply_text = "Market data analysis complete."
            
            resp_model = (agent_response.model.llm if agent_response.model else "vertex") if agent_response else "vertex"
            
            first_token = True
            async for chunk in streaming_service.stream_text(reply_text, conversation_id=conv_id):
                if chunk.chunk_type == "response_token":
                    if first_token:
                        await emitter.streaming_started()
                        first_token = False
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
                    full_response = reply_text
                    total_ms = (time.monotonic() - start_time) * 1000
                    await emitter.done()
                    await safe_send_json({
                        "type": "stream_summary",
                        "ttft_ms": chunk.metadata.get("ttft_ms"),
                        "latency_ms": round(total_ms, 2),
                        "thinking_token_count": 0,
                        "response_token_count": chunk.metadata.get("response_token_count", 0),
                        "model": resp_model,
                        "agent": agent_id,
                    })
                    if "IDBX Data Provenance" not in full_response:
                        full_response += "\n\n**IDBX Data Provenance:** Alpha Vantage Live Feed"
        else:
            # Real token-by-token streaming from Vertex AI
            logger.info(f"[WebSocket:Streaming] 🚀 Starting REAL token-by-token streaming from Vertex AI")
            resp_model = (agent_response.model.llm if agent_response.model else "vertex") if agent_response else "vertex"
            first_token = True
            
            async for chunk in streaming_chunks:
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
                    if first_token:
                        await emitter.streaming_started()
                        first_token = False
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
                    await emitter.done()
                    await safe_send_json({
                        "type": "stream_summary",
                        "ttft_ms": chunk.metadata.get("ttft_ms"),
                        "latency_ms": round(total_ms, 2),
                        "thinking_token_count": chunk.metadata.get("thinking_token_count", 0),
                        "response_token_count": chunk.metadata.get("response_token_count", 0),
                        "model": resp_model,
                        "agent": agent_id,
                    })
                    # Provenance footer
                    if "IDBX Data Provenance" not in full_response:
                        full_response += "\n\n**IDBX Data Provenance:** Alpha Vantage Live Feed"

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
    # PATH B: Non-market — run the SAME coordinator pipeline as non-streaming,
    # then stream the resulting reply text word-by-word.
    # =========================================================================
    else:
        from app.services.aidaan.agents.coordinator.coordinator_agent import coordinator_agent as _coordinator

        agent_response = None
        agent_error: Optional[str] = None
        try:
            agent_response = await _coordinator.handle_message(
                text=user_text,
                conversation_id=conv_id,
                context=context,
                tool_callback=None,
            )
        except Exception as agent_exc:
            logger.error("[WebSocket:Streaming] Coordinator agent failed: %s", agent_exc)
            agent_error = str(agent_exc)

        if agent_error or agent_response is None:
            await safe_send_json({
                "type": "stream_error",
                "error": agent_error or "Agent returned no response",
                "agent": agent_id,
            })
            return None

        reply_text = (agent_response.reply or "").strip()
        if not reply_text:
            reply_text = "I'm here to help with your trading desk operations."

        resp_model = (agent_response.model.llm if agent_response.model else "vertex") if agent_response else "vertex"

        first_token = True
        async for chunk in streaming_service.stream_text(
            reply_text,
            conversation_id=conv_id,
        ):
            if chunk.chunk_type == "response_token":
                if first_token:
                    await emitter.streaming_started()
                    first_token = False
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
                full_response = reply_text
                total_ms = (time.monotonic() - start_time) * 1000
                await emitter.done()
                await safe_send_json({
                    "type": "stream_summary",
                    "ttft_ms": chunk.metadata.get("ttft_ms"),
                    "latency_ms": round(total_ms, 2),
                    "thinking_token_count": 0,
                    "response_token_count": chunk.metadata.get("response_token_count", 0),
                    "model": resp_model,
                    "agent": agent_id,
                })

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


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/aidaan")
async def aidaan_websocket(websocket: WebSocket):
    """
    Real-time WebSocket endpoint for AIDAAN.

    Streaming is opt-in via enable_streaming flag.
    All existing non-streaming functionality is unchanged.

    Streaming message format:
    {"type":"chat","text":"...","enable_streaming":true,"user_id":"...","conversation_id":"..."}
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
                    if not await safe_send_json({"type": "state", "state": "Awakening",
                                                  "detail": "Trader proximity detected."}):
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
                    "type": "state", "state": "Listening", "detail": "Trader input received.",
                }):
                    break

                # DEBUG: Log streaming flag
                logger.info(f"[WebSocket] Message received | enable_streaming={enable_streaming} | user={user_id} | text={user_text[:60]}")

                if enable_streaming:
                    logger.info("[WebSocket] STREAMING mode | user=%s | conv_id=%s | text=%s",
                                user_id, conv_id, user_text[:60])
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
                    if not await safe_send_json({
                        "type": "state", "state": "Thinking", "detail": "Coordinator processing...",
                    }):
                        break
                    await asyncio.sleep(0.1)

                    try:
                        async def tool_pulse(tool_name: str):
                            try:
                                await safe_send_json({
                                    "type": "state", "state": "Thinking",
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
                                "type": "state", "state": "Alert/Busy",
                                "detail": "Risk alert condition detected.",
                            }):
                                break

                        if not await safe_send_json({
                            "type": "chat_reply", "payload": response.model_dump(),
                        }):
                            break
                        dormant_sent = False
                    except Exception as inner_e:
                        logger.error("Agent Processing Error: %s", inner_e)
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
            await safe_send_json({"type": "error", "message": f"WebSocket error: {str(e)}"})
