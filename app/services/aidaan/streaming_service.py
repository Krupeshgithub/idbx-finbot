"""
AIDAAN Vertex AI Streaming Service
====================================
Yeh file Vertex AI ke generate_content_stream() ka wrapper hai.

ARCHITECTURE EXPLANATION (Line by Line):
=========================================

Vertex AI SDK mein 2 methods hain:
  1. generate_content()        → Blocking, pura response ek saath aata hai
  2. generate_content_stream() → Streaming, ek ek token aata hai as model generates

Yeh file Type 2 use karti hai — real streaming from Vertex AI.

IMPORTANT CONSTRAINTS (Jo tumhara code check karke pata chala):
================================================================
1. response_mime_type="application/json" → streaming ke saath INCOMPATIBLE hai
   Isliye streaming path mein JSON mode OFF rahega, text mode ON.

2. MCP Tools ke saath streaming:
   - Tools pehle synchronously execute hote hain (Alpha Vantage calls blocking hain)
   - Sirf FINAL SYNTHESIS step ko stream kar sakte ho
   - Yahi sahi approach hai — tools ka result chunked nahi hota

3. ThreadPoolExecutor mein blocking iterator:
   - generate_content_stream() ek synchronous iterator return karta hai
   - Isko asyncio loop mein directly call nahi kar sakte
   - Solution: run_in_executor mein run karo, asyncio.Queue se chunks pass karo

4. Thinking tokens (gemini-2.5-pro mein available):
   - chunk.candidates[0].content.parts mein thought=True wale parts hote hain
   - Inhe alag stream type se bhejo

HOW STREAMING ACTUALLY WORKS IN THIS CODE:
===========================================

Step 1: User message aata hai WebSocket se
Step 2: StreamingService.stream_response_async() call hoti hai
Step 3: Vertex AI ka generate_content_stream() ek thread mein chalta hai
Step 4: Har chunk asyncio.Queue mein dala jaata hai
Step 5: Async loop queue se chunks uthata hai aur WebSocket pe bhejta hai
Step 6: Frontend pe token by token text appear hota hai

REAL vs FAKE STREAMING:
========================
FAKE: Pura response generate karo, phir character by character split karke bhejo
REAL: Vertex AI MODEL khud token by token generate karta hai, hum sirf relay karte hain

Yeh code REAL streaming karta hai. Vertex AI ka model jaise generate karta hai,
waise hi frontend pe dikhai deta hai.
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

# Dedicated thread pool for streaming calls
# Kyunki generate_content_stream() synchronous iterator hai,
# isko ThreadPoolExecutor mein hi chalana padega
_stream_executor = ThreadPoolExecutor(
    max_workers=10,
    thread_name_prefix="vertex_stream_worker",
)


class StreamChunk:
    """
    Ek single streaming chunk ka representation.
    
    chunk_type values:
      - "thinking_token"  : Model ka internal reasoning (sirf pro models mein)
      - "response_token"  : Actual response text jo user ko dikhana hai
      - "stream_complete" : Streaming khatam, stats ke saath
      - "stream_error"    : Kuch galat hua
      - "tool_start"      : MCP tool call shuru hua
      - "tool_done"       : MCP tool call complete hua
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
    Vertex AI se real streaming karne ka service.
    
    Tumhare existing LLMClient ke parallel chalega — woh non-streaming ke liye,
    yeh streaming ke liye. Dono ek hi Vertex AI client use karte hain.
    """

    def __init__(self) -> None:
        self._client: Optional[genai.Client] = None
        self._initialized = False

    def _ensure_client(self) -> Optional[genai.Client]:
        """
        Vertex AI client initialize karo.
        
        Tumhara existing llm_client.py ka _ensure_client() same logic hai.
        Hum woh reuse karte hain duplicate na ho isliye.
        """
        if self._client is not None:
            return self._client

        try:
            # Import existing LLM client ka client object reuse karo
            # Isse credentials double set nahi honge
            from app.core.llm_client import llm_client
            if llm_client._client is not None:
                self._client = llm_client._client
                logger.info("[StreamingService] Reusing existing Vertex AI client")
                return self._client
        except Exception as e:
            logger.warning("[StreamingService] Could not reuse LLM client: %s", e)

        # Fallback: apna client banao
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
        Streaming ke liye config banao.
        
        CRITICAL DIFFERENCE from non-streaming config:
        ================================================
        Non-streaming mein: response_mime_type="application/json" hota hai
        Streaming mein: response_mime_type NAHI lagani — plain text stream hogi
        
        Kyun? Vertex AI ka streaming API JSON mode ke saath properly work
        nahi karta chunked responses mein — partial JSON parse nahi ho sakta.
        
        Thinking tokens ke liye: thinking_config add karo (sirf pro models mein)
        """
        from app.core.llm_client import llm_client
        
        # System instruction mein language + timestamp inject karo
        # (same as existing LLMClient._apply_language_instruction)
        full_system_instruction = llm_client._apply_language_instruction(system_instruction)

        config_kwargs: Dict[str, Any] = {
            "temperature": settings.VERTEX_AI_TEMPERATURE,
            "max_output_tokens": settings.VERTEX_AI_MAX_OUTPUT_TOKENS,
            "system_instruction": full_system_instruction,
            # NOTE: response_mime_type intentionally OMITTED for streaming
            # JSON mode aur streaming incompatible hain Vertex AI mein
        }

        # Thinking tokens enable karo (sirf gemini-2.5-pro mein kaam karta hai)
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
        Yeh function ek dedicated thread mein chalta hai.
        
        Kyun thread chahiye:
        ====================
        generate_content_stream() ek SYNCHRONOUS generator/iterator return karta hai.
        Isko directly async function mein use nahi kar sakte kyunki yeh blocking hai.
        
        Solution:
        =========
        1. Is function ko ThreadPoolExecutor mein run karo
        2. Har chunk ko asyncio.Queue mein dalo (thread-safe)
        3. Async side queue se read karta rahe
        
        Queue mein kya dalta hai:
        ========================
        - dict: ek chunk ka data
        - None: signal ki streaming khatam ho gayi
        - Exception object: kuch galat hua
        """
        try:
            logger.info("[StreamingService] Starting generate_content_stream in thread | model=%s", model)
            
            # *** YAHI HAI REAL VERTEX AI STREAMING CALL ***
            # generate_content_stream() calls the Vertex AI API and returns
            # chunks as they are generated by the model
            response_stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )

            # Har chunk ko process karo
            # Yeh loop tab tak chalta hai jab tak model tokens generate karta rahe
            for chunk in response_stream:
                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    continue

                for part in candidate.content.parts:
                    # Thinking token check — sirf pro models mein hota hai
                    # part.thought == True matlab yeh model ka internal reasoning hai
                    is_thinking = getattr(part, "thought", False)
                    
                    part_text = getattr(part, "text", "") or ""
                    if not part_text:
                        continue

                    chunk_data = {
                        "is_thinking": is_thinking,
                        "text": part_text,
                    }

                    # Thread-safe tarike se queue mein dalo
                    # loop.call_soon_threadsafe asyncio event loop ko signal karta hai
                    loop.call_soon_threadsafe(queue.put_nowait, chunk_data)

            # None = streaming complete signal
            loop.call_soon_threadsafe(queue.put_nowait, None)
            logger.info("[StreamingService] Stream completed in thread | model=%s", model)

        except Exception as exc:
            # Exception ko queue mein dalo taaki async side handle kar sake
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
        Vertex AI se real streaming — main async generator.
        
        Yeh function yield karta hai StreamChunk objects as tokens arrive from Vertex AI.
        
        Usage:
        ======
        async for chunk in streaming_service.stream_response_async(prompt):
            await websocket.send_json(chunk.to_dict())
        
        What happens internally:
        ========================
        1. DLP redaction on prompt
        2. Queue + thread setup
        3. Thread mein generate_content_stream() call
        4. Queue se chunks read karke yield karo
        5. Completion stats yield karo
        
        Args:
            prompt: User ka message ya formatted prompt
            model_name: Vertex AI model name (default: settings.VERTEX_AI_MODEL_NAME)
            system_instruction: System prompt
            enable_thinking: Thinking tokens ON/OFF (sirf pro models mein)
            conversation_id: For logging
            username: For DLP/audit
        
        Yields:
            StreamChunk objects with types: thinking_token, response_token, stream_complete, stream_error
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

        # DLP redaction — same as existing code
        try:
            prompt = await asyncio.get_event_loop().run_in_executor(
                _stream_executor, dlp_client.redact_pii, prompt
            )
        except Exception as dlp_err:
            logger.warning("[StreamingService] DLP redaction failed (continuing): %s", dlp_err)

        # Config build karo — WITHOUT JSON mode
        config = self._build_streaming_config(
            system_instruction=system_instruction,
            enable_thinking=enable_thinking,
        )

        # Queue banao thread aur async ke beech communication ke liye
        # maxsize=100 taaki memory overflow na ho
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        loop = asyncio.get_event_loop()

        # Thread mein streaming call shuru karo
        # run_in_executor non-blocking hai — async loop block nahi hoga
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

        # Async loop — queue se chunks read karo aur yield karo
        try:
            while True:
                # Queue se next item lo — max 60 seconds wait
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

                # Exception = kuch galat hua thread mein
                if isinstance(item, Exception):
                    error_msg = str(item)
                    logger.error("[StreamingService] Stream error received: %s", error_msg)
                    yield StreamChunk(
                        chunk_type="stream_error",
                        content=f"Streaming error: {error_msg}",
                    )
                    break

                # Normal chunk — process karo
                is_thinking = item.get("is_thinking", False)
                text = item.get("text", "")

                if not text:
                    continue

                # TTFT (Time To First Token) track karo
                if first_token_time is None:
                    first_token_time = time.monotonic()
                    ttft_ms = (first_token_time - start_time) * 1000
                    logger.info("[StreamingService] First token received | TTFT=%.1fms | model=%s", ttft_ms, model)

                if is_thinking:
                    # Thinking token — model ka internal reasoning
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
                    # Response token — actual answer jo user ko dikhana hai
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
            # Thread future cancel karo agar abhi bhi chal raha ho
            if not future.done():
                future.cancel()

        # Final stats chunk — streaming khatam hone ke baad
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
        MCP tools execute hone ke BAAD streaming synthesis.
        
        Yeh tumhare use case ke liye sabse important method hai.
        
        Flow:
        =====
        1. Alpha Vantage / MCP tools already call ho chuke hain (non-streaming)
        2. Tool results available hain
        3. Ab Vertex AI ko in results ke saath final answer stream karo
        
        Kyun yeh pattern sahi hai:
        ==========================
        Alpha Vantage API khud streaming nahi karta — JSON response deta hai.
        Tools ka result pehle fully aana chahiye, phir synthesis stream hoti hai.
        
        Contents format:
        ================
        Hum tool results ko conversation history format mein dete hain.
        Yeh wahi format hai jo tumhara existing _generate_json_with_mcp_tools uses.
        
        Args:
            tool_results: List of {tool_name, result} dicts from MCP tools
            original_prompt: User ka original question
            system_instruction: System prompt
            model_name: Model to use for synthesis
            conversation_id: For tracking
            username: For DLP
        
        Yields:
            StreamChunk objects — same as stream_response_async
        """
        model = model_name or settings.VERTEX_AI_MODEL_NAME

        # Tool results ko readable format mein convert karo
        tool_context_parts = []
        for i, tool_result in enumerate(tool_results):
            tool_name = tool_result.get("tool_name", f"tool_{i}")
            result = tool_result.get("result", {})
            tool_context_parts.append(
                f"Tool: {tool_name}\nResult: {json.dumps(result, ensure_ascii=False, default=str)[:2000]}"
            )

        tool_context = "\n\n".join(tool_context_parts)

        # Final synthesis prompt banao
        synthesis_prompt = (
            f"{original_prompt}\n\n"
            f"=== TOOL RESULTS ===\n{tool_context}\n\n"
            f"=== INSTRUCTIONS ===\n"
            f"Upar diye gaye tool results ke basis par professional trader response do.\n"
            f"Structure: [Direct Answer] → [Market Insight] → [Trade Implication] → [Optional Follow-up]\n"
            f"CRITICAL: Return ONLY plain text response — no JSON, no markdown code blocks.\n"
            f"Professional trading desk language use karo."
        )

        logger.info(
            "[StreamingService] Starting post-tool synthesis streaming | model=%s | tools=%d | conv_id=%s",
            model, len(tool_results), conversation_id
        )

        # stream_response_async ko delegate karo
        async for chunk in self.stream_response_async(
            prompt=synthesis_prompt,
            model_name=model,
            system_instruction=system_instruction,
            enable_thinking=False,  # Post-tool synthesis mein thinking OFF
            conversation_id=conversation_id,
            username=username,
        ):
            yield chunk


# Global singleton instance
streaming_service = StreamingService()
