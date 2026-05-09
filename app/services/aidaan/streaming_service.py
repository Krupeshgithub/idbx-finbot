"""
AIDAAN Vertex AI Streaming Service
====================================
Wrapper around Vertex AI's generate_content_stream() for real token-by-token streaming.

ARCHITECTURE:
=============
Vertex AI SDK exposes two generation methods:
  1. generate_content()        -> Blocking, returns the full response at once
  2. generate_content_stream() -> Streaming, yields one chunk at a time as the model generates

This file uses method 2 — real streaming from Vertex AI.

IMPORTANT CONSTRAINTS:
======================
1. response_mime_type="application/json" is INCOMPATIBLE with streaming.
   The streaming path intentionally omits JSON mode and uses plain text.

2. MCP Tools + Streaming:
   - Tools execute synchronously first (Alpha Vantage calls are blocking)
   - Only the FINAL SYNTHESIS step is streamed
   - This is the correct approach — tool results are not chunked

3. ThreadPoolExecutor for blocking iterator:
   - generate_content_stream() returns a synchronous iterator
   - It cannot be called directly inside an async function (it blocks the event loop)
   - Solution: run it in a ThreadPoolExecutor, pass chunks via asyncio.Queue

4. Thinking tokens (available in gemini-2.5-pro):
   - chunk.candidates[0].content.parts may contain parts with thought=True
   - These are streamed separately as "thinking_token" type

HOW STREAMING WORKS IN THIS CODE:
===================================
Step 1: User message arrives via WebSocket
Step 2: StreamingService.stream_response_async() is called
Step 3: Vertex AI's generate_content_stream() runs in a dedicated thread
Step 4: Each chunk is placed into an asyncio.Queue
Step 5: The async loop reads chunks from the queue and sends them over WebSocket
Step 6: The frontend receives and renders tokens one by one

REAL vs FAKE STREAMING:
========================
FAKE: Generate the full response, then split it character by character and send
REAL: The Vertex AI model generates tokens incrementally; we relay them as they arrive

This code performs REAL streaming.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Dict, List, Optional

from google import genai
from google.genai import types

from app.core.config.settings import settings
from app.core.dlp_client import dlp_client

logger = logging.getLogger(__name__)

# Dedicated thread pool for streaming calls.
# generate_content_stream() is a synchronous iterator and must run in a thread.
_stream_executor = ThreadPoolExecutor(
    max_workers=10,
    thread_name_prefix="vertex_stream_worker",
)


class StreamChunk:
    """
    Represents a single streaming chunk from Vertex AI.

    chunk_type values:
      - "thinking_token"  : Model's internal reasoning (pro models only)
      - "response_token"  : Actual response text to display to the user
      - "stream_complete" : Streaming finished, includes final stats
      - "stream_error"    : An error occurred during streaming
      - "tool_start"      : An MCP tool call has started
      - "tool_done"       : An MCP tool call has completed
    """
    def __init__(
        self,
        chunk_type: str,
        content: str = "",
        token_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunk_type = chunk_type
        self.content = content
        self.token_count = token_count
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.chunk_type,
            "content": self.content,
            "token_count": self.token_count,
            **self.metadata,
        }


class StreamingService:
    """
    Service for real token-by-token streaming from Vertex AI.

    Runs in parallel with the existing LLMClient — that client handles
    non-streaming calls, this service handles streaming. Both share the
    same underlying Vertex AI client instance.
    """

    def __init__(self) -> None:
        self._client: Optional[genai.Client] = None
        self._initialized = False

    def _ensure_client(self) -> Optional[genai.Client]:
        """
        Initialize or reuse the Vertex AI client.

        Reuses the existing llm_client's initialized client to avoid
        double credential setup.
        """
        if self._client is not None:
            return self._client

        try:
            from app.core.llm_client import llm_client
            if llm_client._client is not None:
                self._client = llm_client._client
                logger.info("[StreamingService] Reusing existing Vertex AI client")
                return self._client
        except Exception as e:
            logger.warning("[StreamingService] Could not reuse LLM client: %s", e)

        # Fallback: initialize a new client
        try:
            import os
            has_adc = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
            can_express = bool(settings.VERTEX_AI_USE_EXPRESS_MODE and settings.GOOGLE_API_KEY)

            if not has_adc and not can_express:
                logger.error("[StreamingService] No Vertex AI credentials found")
                return None

            client_kwargs: Dict[str, Any] = {
                "vertexai": True,
                "http_options": types.HttpOptions(api_version=settings.VERTEX_AI_API_VERSION),
            }

            if has_adc:
                client_kwargs["project"] = settings.GOOGLE_CLOUD_PROJECT
                client_kwargs["location"] = settings.GOOGLE_CLOUD_LOCATION
            else:
                client_kwargs["api_key"] = settings.GOOGLE_API_KEY

            self._client = genai.Client(**client_kwargs)
            logger.info("[StreamingService] Vertex AI streaming client initialized")
            return self._client
        except Exception as exc:
            logger.error("[StreamingService] Client initialization failed: %s", exc)
            return None

    def _build_streaming_config(
        self,
        system_instruction: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> types.GenerateContentConfig:
        """
        Build the GenerateContentConfig for a streaming call.

        CRITICAL DIFFERENCE from non-streaming config:
        -----------------------------------------------
        Non-streaming uses: response_mime_type="application/json"
        Streaming intentionally omits response_mime_type (plain text stream)

        Reason: Vertex AI's streaming API does not work correctly with JSON mode
        because partial JSON chunks cannot be parsed incrementally.

        Thinking tokens are enabled via thinking_config (pro models only).
        """
        from app.core.llm_client import llm_client

        # Inject language instruction and timestamp (same as LLMClient._apply_language_instruction)
        full_system_instruction = llm_client._apply_language_instruction(system_instruction)

        config_kwargs: Dict[str, Any] = {
            "temperature": settings.VERTEX_AI_TEMPERATURE,
            "max_output_tokens": settings.VERTEX_AI_MAX_OUTPUT_TOKENS,
            "system_instruction": full_system_instruction,
            # NOTE: response_mime_type intentionally omitted — JSON mode is incompatible with streaming
        }

        if enable_thinking:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True,
            )
            logger.info("[StreamingService] Thinking tokens enabled")

        return types.GenerateContentConfig(**config_kwargs)

    def _run_stream_in_thread(
        self,
        client: genai.Client,
        model: str,
        contents: Any,
        config: types.GenerateContentConfig,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Runs generate_content_stream() inside a dedicated thread.

        Why a thread is required:
        --------------------------
        generate_content_stream() returns a SYNCHRONOUS generator/iterator.
        Calling it directly in an async function would block the event loop.

        Solution:
        ---------
        1. Run this function in a ThreadPoolExecutor
        2. Push each chunk into an asyncio.Queue (thread-safe via call_soon_threadsafe)
        3. The async side reads from the queue and yields chunks

        Queue sentinel values:
        ----------------------
        - dict  : a normal chunk with keys "is_thinking" and "text"
        - None  : signals that streaming has completed
        - Exception object : signals that an error occurred in the thread
        """
        try:
            logger.info("[StreamingService] Starting generate_content_stream in thread | model=%s", model)

            # THE REAL VERTEX AI STREAMING CALL
            # generate_content_stream() calls the Vertex AI API and yields
            # chunks as the model generates them token by token.
            response_stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )

            for chunk in response_stream:
                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    continue

                for part in candidate.content.parts:
                    # part.thought == True means this is the model's internal reasoning
                    # Only available in pro models with thinking enabled
                    is_thinking = getattr(part, "thought", False)

                    part_text = getattr(part, "text", "") or ""
                    if not part_text:
                        continue

                    chunk_data = {
                        "is_thinking": is_thinking,
                        "text": part_text,
                    }

                    # Thread-safe: push to asyncio queue from a non-async thread
                    loop.call_soon_threadsafe(queue.put_nowait, chunk_data)

            # None = streaming complete sentinel
            loop.call_soon_threadsafe(queue.put_nowait, None)
            logger.info("[StreamingService] Stream completed in thread | model=%s", model)

        except Exception as exc:
            # Push the exception into the queue so the async side can handle it
            logger.error("[StreamingService] Stream error in thread: %s", exc)
            loop.call_soon_threadsafe(queue.put_nowait, exc)

    async def stream_response_async(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        system_instruction: Optional[str] = None,
        enable_thinking: bool = False,
        conversation_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Main async generator for real token-by-token streaming from Vertex AI.

        Yields StreamChunk objects as tokens arrive from the model.

        Usage:
        ------
        async for chunk in streaming_service.stream_response_async(prompt):
            await websocket.send_json(chunk.to_dict())

        Internal flow:
        --------------
        1. DLP redaction on the prompt
        2. Set up asyncio.Queue and ThreadPoolExecutor thread
        3. Thread calls generate_content_stream() and pushes chunks to queue
        4. Async loop reads from queue and yields StreamChunk objects
        5. Final stream_complete chunk includes latency stats

        Args:
            prompt: User message or formatted prompt string
            model_name: Vertex AI model name (defaults to settings.VERTEX_AI_MODEL_NAME)
            system_instruction: System prompt
            enable_thinking: Enable thinking tokens (pro models only)
            conversation_id: Used for logging and tracing
            username: Used for DLP and audit

        Yields:
            StreamChunk with types: thinking_token, response_token, stream_complete, stream_error
        """
        model = model_name or settings.VERTEX_AI_MODEL_NAME
        start_time = time.monotonic()
        first_token_time: Optional[float] = None
        thinking_token_count = 0
        response_token_count = 0
        full_response_text = ""

        client = self._ensure_client()
        if client is None:
            yield StreamChunk(
                chunk_type="stream_error",
                content="Vertex AI client not initialized. Check credentials.",
            )
            return

        # DLP redaction — same as existing non-streaming path
        try:
            prompt = await asyncio.get_event_loop().run_in_executor(
                _stream_executor, dlp_client.redact_pii, prompt
            )
        except Exception as dlp_err:
            logger.warning("[StreamingService] DLP redaction failed (continuing): %s", dlp_err)

        # Build config WITHOUT JSON mode
        config = self._build_streaming_config(
            system_instruction=system_instruction,
            enable_thinking=enable_thinking,
        )

        # Queue for thread <-> async communication
        # maxsize=100 prevents memory overflow if the consumer is slow
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        loop = asyncio.get_event_loop()

        # Start the streaming thread (non-blocking — event loop is not blocked)
        future = loop.run_in_executor(
            _stream_executor,
            self._run_stream_in_thread,
            client,
            model,
            prompt,
            config,
            queue,
            loop,
        )

        logger.info(
            "[StreamingService] Streaming started | model=%s | conv_id=%s | thinking=%s",
            model, conversation_id, enable_thinking
        )

        try:
            while True:
                # Wait for the next chunk from the thread (60s timeout)
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=60.0)
                except asyncio.TimeoutError:
                    logger.error("[StreamingService] Queue timeout — stream stalled")
                    yield StreamChunk(
                        chunk_type="stream_error",
                        content="Stream timeout — Vertex AI did not respond in time.",
                    )
                    break

                # None = streaming complete
                if item is None:
                    break

                # Exception = error in the thread
                if isinstance(item, Exception):
                    error_msg = str(item)
                    logger.error("[StreamingService] Stream error received: %s", error_msg)
                    yield StreamChunk(
                        chunk_type="stream_error",
                        content=f"Streaming error: {error_msg}",
                    )
                    break

                # Normal chunk
                is_thinking = item.get("is_thinking", False)
                text = item.get("text", "")

                if not text:
                    continue

                # Track Time To First Token (TTFT)
                if first_token_time is None:
                    first_token_time = time.monotonic()
                    ttft_ms = (first_token_time - start_time) * 1000
                    logger.info("[StreamingService] First token received | TTFT=%.1fms | model=%s", ttft_ms, model)

                if is_thinking:
                    # Thinking token — model's internal reasoning, shown separately in UI
                    thinking_token_count += 1
                    yield StreamChunk(
                        chunk_type="thinking_token",
                        content=text,
                        token_count=thinking_token_count,
                        metadata={
                            "elapsed_ms": round((time.monotonic() - start_time) * 1000, 2),
                        },
                    )
                else:
                    # Response token — the actual answer text
                    response_token_count += 1
                    full_response_text += text
                    yield StreamChunk(
                        chunk_type="response_token",
                        content=text,
                        token_count=response_token_count,
                        metadata={
                            "elapsed_ms": round((time.monotonic() - start_time) * 1000, 2),
                        },
                    )

        except Exception as exc:
            logger.error("[StreamingService] Async streaming loop error: %s", exc)
            yield StreamChunk(
                chunk_type="stream_error",
                content=f"Unexpected streaming error: {exc}",
            )
        finally:
            # Cancel the thread future if it is still running
            if not future.done():
                future.cancel()

        # Final stats chunk — sent after all tokens have been yielded
        total_elapsed_ms = (time.monotonic() - start_time) * 1000
        ttft_final = (
            round((first_token_time - start_time) * 1000, 2)
            if first_token_time
            else None
        )

        logger.info(
            "[StreamingService] Stream complete | model=%s | ttft_ms=%s | total_ms=%.1f | "
            "thinking_tokens=%d | response_tokens=%d | conv_id=%s",
            model,
            ttft_final,
            total_elapsed_ms,
            thinking_token_count,
            response_token_count,
            conversation_id,
        )

        yield StreamChunk(
            chunk_type="stream_complete",
            content=full_response_text,
            token_count=response_token_count,
            metadata={
                "ttft_ms": ttft_final,
                "total_latency_ms": round(total_elapsed_ms, 2),
                "thinking_token_count": thinking_token_count,
                "response_token_count": response_token_count,
                "model": model,
                "conversation_id": conversation_id,
            },
        )

    async def stream_after_tools(
        self,
        *,
        tool_results: List[Dict[str, Any]],
        original_prompt: str,
        system_instruction: Optional[str] = None,
        model_name: Optional[str] = None,
        conversation_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream the final synthesis AFTER MCP tools have already executed.

        This is the correct pattern for Type 2 (Post-Tool) streaming:

        Flow:
        -----
        1. Alpha Vantage / MCP tools have already been called (non-streaming, blocking)
        2. Tool results are available as structured dicts
        3. Vertex AI streams the final synthesis using those results

        Why this pattern is correct:
        ----------------------------
        Alpha Vantage does not stream — it returns a complete JSON response.
        Tools must finish before synthesis can begin.
        Only the synthesis step is streamed to the user.

        Contents format:
        ----------------
        Tool results are formatted as a readable context block and injected
        into the synthesis prompt. This mirrors the format used by
        _generate_json_with_mcp_tools in the non-streaming path.

        Args:
            tool_results: List of {"tool_name": str, "result": dict} from MCP tools
            original_prompt: The user's original question
            system_instruction: System prompt for the synthesis model
            model_name: Model to use for synthesis
            conversation_id: For tracing and logging
            username: For DLP

        Yields:
            StreamChunk objects — same types as stream_response_async
        """
        model = model_name or settings.VERTEX_AI_MODEL_NAME

        # Format tool results into a readable context block
        tool_context_parts = []
        for i, tool_result in enumerate(tool_results):
            tool_name = tool_result.get("tool_name", f"tool_{i}")
            result = tool_result.get("result", {})
            tool_context_parts.append(
                f"Tool: {tool_name}\nResult: {json.dumps(result, ensure_ascii=False, default=str)[:2000]}"
            )

        tool_context = "\n\n".join(tool_context_parts)

        # Build the synthesis prompt
        synthesis_prompt = (
            f"{original_prompt}\n\n"
            f"=== TOOL RESULTS ===\n{tool_context}\n\n"
            f"=== INSTRUCTIONS ===\n"
            f"Based on the tool results above, provide a professional trader response.\n"
            f"Structure: [Direct Answer] -> [Market Insight] -> [Trade Implication] -> [Optional Follow-up]\n"
            f"CRITICAL: Return ONLY plain text — no JSON, no markdown code blocks.\n"
            f"Use professional trading desk language."
        )

        logger.info(
            "[StreamingService] Starting post-tool synthesis streaming | model=%s | tools=%d | conv_id=%s",
            model, len(tool_results), conversation_id
        )

        # Delegate to stream_response_async
        async for chunk in self.stream_response_async(
            prompt=synthesis_prompt,
            model_name=model,
            system_instruction=system_instruction,
            enable_thinking=False,  # Thinking OFF for post-tool synthesis
            conversation_id=conversation_id,
            username=username,
        ):
            yield chunk


# Global singleton instance
streaming_service = StreamingService()
