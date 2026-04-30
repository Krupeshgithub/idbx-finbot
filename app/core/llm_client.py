"""
AIDAAN Vertex AI LLM Client
===========================
One place to configure, call, and debug all Vertex AI interactions.

HOW TO USE:
    from app.core.llm_client import llm_client

    result = await llm_client.generate_json(prompt)
    text = await llm_client.generate_text(prompt)

HOW TO SWAP MODELS:
    - Change VERTEX_AI_MODEL_NAME in settings.py for default flows
    - Change VERTEX_AI_REASONING_MODEL_NAME for heavier analytical flows
    - Or pass model_override="gemini-2.5-pro" for one-off calls
"""
import asyncio
import datetime
import hashlib
import json
import logging
import os
import re
import shlex
import sys
import time
from contextlib import AsyncExitStack, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from uuid import uuid4

from google import genai
from google.genai import types

from app.core.cache import cache
from app.core.config.settings import settings
from app.core.dlp_client import dlp_client
from app.core.audit_logger import audit_logger

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(
    max_workers=settings.VERTEX_AI_MAX_WORKERS,
    thread_name_prefix="vertex_llm_worker",
)


class LLMClient:
    """
    Generic async wrapper around the Google Gen AI SDK on Vertex AI.
    Single instance used by all agents via dependency injection or singleton.
    """

    def __init__(self) -> None:
        self._default_model_name = settings.VERTEX_AI_MODEL_NAME
        self._reasoning_model_name = settings.VERTEX_AI_REASONING_MODEL_NAME
        self._mcp_tools = []
        self._client: Optional[genai.Client] = None
        self._vertex_disabled_reason: Optional[str] = None
        self._runtime_mode: str = "uninitialized"
        self._temp_credential_file: Optional[Path] = None
        self._usage_stats = {
            "generate_json": 0,
            "generate_json_sync": 0,
            "generate_text": 0,
        }
        self._mcp_inprocess_server = None
        self._vertex_concurrency = asyncio.Semaphore(settings.VERTEX_AI_MAX_CONCURRENT_REQUESTS)
        self._context_caches: Dict[str, Any] = {}  # key: hash(model+instruction), value: cached_content_name

        self._ensure_client()

    def _ensure_client(self) -> Optional[genai.Client]:
        if self._client is not None:
            return self._client

        has_standard_vertex_auth = bool(
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
        can_try_express_mode = bool(
            settings.VERTEX_AI_USE_EXPRESS_MODE and settings.GOOGLE_API_KEY
        )

        if not has_standard_vertex_auth and not can_try_express_mode:
            self._vertex_disabled_reason = (
                "Vertex AI is not configured. Provide service-account/ADC credentials for standard Vertex AI "
                "or set GOOGLE_API_KEY for Vertex Express Mode."
            )
            logger.warning("[LLMClient] %s", self._vertex_disabled_reason)
            return None

        try:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            client_kwargs: Dict[str, Any] = {
                "vertexai": True,
                "http_options": types.HttpOptions(api_version=settings.VERTEX_AI_API_VERSION),
            }

            if has_standard_vertex_auth:
                if not settings.GOOGLE_CLOUD_PROJECT:
                    self._vertex_disabled_reason = (
                        "GOOGLE_CLOUD_PROJECT is required for standard Vertex AI authentication."
                    )
                    logger.warning("[LLMClient] %s", self._vertex_disabled_reason)
                    return None
                client_kwargs["project"] = settings.GOOGLE_CLOUD_PROJECT
                client_kwargs["location"] = settings.GOOGLE_CLOUD_LOCATION
                self._runtime_mode = "vertex_standard"
            else:
                client_kwargs["api_key"] = settings.GOOGLE_API_KEY
                self._runtime_mode = "vertex_express"

            self._client = genai.Client(**client_kwargs)
            self._vertex_disabled_reason = None
            logger.info(
                "[LLMClient] Initialized Vertex AI client | mode=%s | project=%s | location=%s | default_model=%s | reasoning_model=%s | api_version=%s",
                self._runtime_mode,
                settings.GOOGLE_CLOUD_PROJECT,
                settings.GOOGLE_CLOUD_LOCATION,
                self._default_model_name,
                self._reasoning_model_name,
                settings.VERTEX_AI_API_VERSION,
            )
            return self._client
        except Exception as exc:
            self._vertex_disabled_reason = f"Vertex AI client initialization failed: {exc}"
            self._runtime_mode = "unavailable"
            logger.error("[LLMClient] %s", self._vertex_disabled_reason)
            return None

    def _vertex_error_payload(self) -> Dict[str, Any]:
        reason = self._vertex_disabled_reason or "Vertex AI is not available."
        return {"error": reason}

    @staticmethod
    def _new_trace_id(prefix: str) -> str:
        """
        Build a compact trace id so multi-stage latency logs can be correlated.
        """
        return f"{prefix}-{uuid4().hex[:8]}"

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        """
        Convert a monotonic start time into integer milliseconds.
        """
        return int((time.monotonic() - start) * 1000)

    async def _run_generate_content(
        self,
        client: genai.Client,
        model: str,
        contents: Any,
        config: types.GenerateContentConfig,
        *,
        trace_id: Optional[str] = None,
        stage: str = "vertex_generate_content",
        retries: int = 3,
    ) -> Any:
        last_error = None
        for attempt in range(retries):
            started = time.monotonic()
            logger.info(f"[TIMING] Gemini API call starting | stage={stage} | model={model} | attempt={attempt+1}")
            try:
                async with self._vertex_concurrency:
                    # Add timeout wrapper to prevent indefinite hangs
                    response = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            _executor,
                            lambda: client.models.generate_content(
                                model=model,
                                contents=contents,
                                config=config,
                            ),
                        ),
                        timeout=settings.VERTEX_AI_REQUEST_TIMEOUT_SECONDS
                    )
                elapsed = self._elapsed_ms(started)
                logger.info(
                    "[LLMClient][%s] %s completed | model=%s | latency_ms=%s | attempt=%s",
                    trace_id or "no-trace",
                    stage,
                    model,
                    elapsed,
                    attempt + 1,
                )
                logger.info(f"[TIMING] Gemini API call completed | stage={stage} | elapsed={elapsed/1000:.3f}s")
                return response
            except asyncio.TimeoutError:
                elapsed = self._elapsed_ms(started)
                last_error = TimeoutError(f"Gemini API call timed out after {elapsed/1000:.1f}s")
                logger.error(
                    "[LLMClient][%s] %s TIMEOUT | model=%s | elapsed_ms=%s | attempt=%s/%s",
                    trace_id or "no-trace",
                    stage,
                    model,
                    elapsed,
                    attempt + 1,
                    retries,
                )
                if attempt < retries - 1:
                    delay = (attempt + 1) * 5
                    logger.info("[LLMClient][%s] Retrying after timeout in %0.2fs", trace_id or "no-trace", delay)
                    await asyncio.sleep(delay)
                else:
                    raise last_error
            except Exception as exc:
                last_error = exc
                is_429 = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                if is_429 and attempt < retries - 1:
                    # Intra-step backoff to handle transient quota spikes
                    delay = (attempt + 1) * 10 + (time.time() % 5)
                    logger.warning(
                        "[LLMClient][%s] %s hit 429 | model=%s | retrying in %0.2fs",
                        trace_id or "no-trace",
                        stage,
                        model,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise exc
        raise last_error

    @asynccontextmanager
    async def _mcp_session_context(
        self, *, trace_id: Optional[str] = None
    ) -> AsyncIterator[Any]:
        """
        Provide an MCP server/session scoped to a single request/tool loop.
        This prevents different conversations from sharing the same stdio session.
        """
        if settings.MCP_TRANSPORT == "inprocess":
            if self._mcp_inprocess_server is not None:
                yield self._mcp_inprocess_server
                return
            try:
                started = time.monotonic()
                from app.services.aidaan.mcp.server import mcp as market_mcp

                self._mcp_inprocess_server = market_mcp
                logger.info(
                    "[LLMClient][%s] Using in-process MCP server | latency_ms=%s",
                    trace_id or "no-trace",
                    self._elapsed_ms(started),
                )
                yield self._mcp_inprocess_server
                return
            except Exception as exc:
                logger.error("[LLMClient] In-process MCP server import failed: %s", exc)
                yield None
                return

        try:
            started = time.monotonic()
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            raw_args: Any = settings.MCP_SERVER_ARGS
            if isinstance(raw_args, str):
                mcp_args = shlex.split(raw_args) if raw_args.strip() else []
            elif isinstance(raw_args, (list, tuple)):
                mcp_args = [str(item) for item in raw_args]
            else:
                mcp_args = [str(raw_args)]

            command = settings.MCP_SERVER_COMMAND
            # Treat generic python launchers as "current interpreter" to keep MCP in the same venv/container.
            if command in {"python", "python3"}:
                command = sys.executable
            if Path(command).name.startswith("python") and "-u" not in mcp_args:
                mcp_args = ["-u", *mcp_args]

            server_params = StdioServerParameters(
                command=command,
                args=mcp_args,
            )

            async with AsyncExitStack() as exit_stack:
                read, write = await exit_stack.enter_async_context(stdio_client(server_params))
                session = await exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                logger.info(
                    "[LLMClient][%s] Request-scoped MCP session initialized | latency_ms=%s",
                    trace_id or "no-trace",
                    self._elapsed_ms(started),
                )
                yield session
        except Exception as exc:
            logger.error("[LLMClient] Failed to initialize MCP session: %s", exc)
            yield None

    async def _list_mcp_tools(self, session: Any, *, trace_id: Optional[str] = None):
        """
        Fetch tools from the provided MCP session/server.
        """
        if self._mcp_tools:
            return self._mcp_tools

        started = time.monotonic()
        if not session:
            return []

        try:
            if settings.MCP_TRANSPORT == "inprocess":
                self._mcp_tools = await session.list_tools()
            else:
                tools_resp = await session.list_tools()
                self._mcp_tools = tools_resp.tools
            logger.info(
                "[LLMClient][%s] Discovered %s tools via MCP (%s) | latency_ms=%s",
                trace_id or "no-trace",
                len(self._mcp_tools),
                settings.MCP_TRANSPORT,
                self._elapsed_ms(started),
            )
            return self._mcp_tools
        except Exception as exc:
            logger.error("[LLMClient] MCP tool discovery failed: %s", exc)
            return []

    async def _execute_mcp_tool(
        self,
        session: Any,
        name: str,
        arguments: Dict[str, Any],
        *,
        trace_id: Optional[str] = None,
    ) -> Any:
        """
        Execute a specific tool via the provided MCP session.
        """
        started = time.monotonic()
        if not session:
            return {"error": "MCP session not available."}

        try:
            if settings.MCP_TRANSPORT == "inprocess":
                content, meta = await session.call_tool(name, arguments)
                if isinstance(meta, dict) and isinstance(meta.get("result"), dict):
                    logger.info(
                        "[LLMClient][%s] MCP tool completed | tool=%s | latency_ms=%s",
                        trace_id or "no-trace",
                        name,
                        self._elapsed_ms(started),
                    )
                    return meta["result"]
                normalized = self._normalize_mcp_result(content)
                logger.info(
                    "[LLMClient][%s] MCP tool completed | tool=%s | latency_ms=%s",
                    trace_id or "no-trace",
                    name,
                    self._elapsed_ms(started),
                )
                return normalized

            result = await session.call_tool(name, arguments)
            normalized = self._normalize_mcp_result(result.content)
            logger.info(
                "[LLMClient][%s] MCP tool completed | tool=%s | latency_ms=%s",
                trace_id or "no-trace",
                name,
                self._elapsed_ms(started),
            )
            return normalized
        except Exception as exc:
            logger.error("[LLMClient] MCP tool execution failed (%s): %s", name, exc)
            return {"error": f"Error executing tool '{name}': {exc}"}

    def _normalize_mcp_result(self, content: Any) -> Any:
        """
        Convert MCP content blocks into plain Python structures when possible.
        """
        if not isinstance(content, list):
            return content

        normalized = []
        for item in content:
            text = getattr(item, "text", None)
            if text is None and isinstance(item, dict):
                text = item.get("text")

            if text is None:
                normalized.append(item)
                continue

            try:
                # Security: Limit JSON result size to 32KB to avoid TPM overflow
                if len(text) > 32768:
                    logger.warning("[LLMClient] Truncating large tool result (%s chars)", len(text))
                    text = text[:32000] + "... [Truncated for Token Optimization]"
                
                normalized.append(json.loads(text))
            except Exception:
                normalized.append(text)

        return normalized[0] if len(normalized) == 1 else normalized

    @staticmethod
    def _tool_call_signature(name: str, arguments: Dict[str, Any]) -> str:
        """
        Build a stable signature for duplicate MCP tool-call detection.
        """
        return json.dumps(
            {"name": name, "arguments": arguments},
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )

    async def call_mcp_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Public wrapper for explicit MCP tool execution from agents/services.
        """
        trace_id = self._new_trace_id("mcp")
        async with self._mcp_session_context(trace_id=trace_id) as session:
            return await self._execute_mcp_tool(
                session,
                name,
                arguments,
                trace_id=trace_id,
            )

    def _cache_key(
        self,
        prefix: str,
        model_name: str,
        payload: str,
        *,
        schema: Optional[Any] = None,
        system_instruction: Optional[str] = None,
        use_mcp_tools: bool = False,
    ) -> str:
        cache_payload = json.dumps(
            {
                "payload": payload,
                "schema": schema,
                "system_instruction": system_instruction,
                "use_mcp_tools": use_mcp_tools,
            },
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
        return f"llm:{prefix}:{model_name}:{digest}"

    def _log_usage(self, method: str, prompt: str, cached: bool = False) -> None:
        self._usage_stats[method] += 1
        logger.info(
            "[LLMClient] %s call #%s | approx_chars=%s | cached=%s",
            method,
            self._usage_stats[method],
            len(prompt),
            cached,
        )

    def get_usage_snapshot(self) -> Dict[str, int]:
        return dict(self._usage_stats)

    def get_default_model_name(self) -> str:
        return self._default_model_name

    def get_reasoning_model_name(self) -> str:
        return self._reasoning_model_name

    def _apply_language_instruction(self, system_instruction: Optional[str] = None) -> str:
        from app.core.prompts import Prompts
        lang_instruction = Prompts.MULTI_LANGUAGE_INSTRUCTION
        if settings.SUPPORTED_LANGUAGES and settings.SUPPORTED_LANGUAGES.lower() not in ["all", "any", "unrestricted"]:
            lang_instruction += f"\nNote: Try to prioritize supporting these specific languages if queried: {settings.SUPPORTED_LANGUAGES}"

        # Phase 4 Fix: Add explicit Unicode handling instruction
        unicode_instruction = (
            "\n\nCRITICAL JSON ENCODING RULE:\n"
            "When generating JSON responses with non-ASCII characters (Hindi, Gujarati, Marathi, etc.), "
            "you MUST use proper UTF-8 encoding. Do NOT use incomplete Unicode escape sequences like \\u092. "
            "Either use complete \\uXXXX escapes (4 hex digits) or use raw UTF-8 characters directly. "
            "Prefer raw UTF-8 characters for better readability."
        )

        timestamp_instruction = self._build_timestamp_instruction()
        instruction_parts = [part for part in [system_instruction, timestamp_instruction, lang_instruction, unicode_instruction] if part]
        return "\n\n".join(instruction_parts)

    @staticmethod
    def _build_timestamp_instruction() -> str:
        """
        Inject a rounded server timestamp into the hidden system instruction.
        
        Rounding to the nearest hour ensures the system instruction remains static 
        for an hour, enabling high-performance Prompt Caching while maintaining
        reasonable time-awareness for the model.
        """
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        # Round to nearest hour: replace minute, second, microsecond with 0
        rounded_utc = now_utc.replace(minute=0, second=0, microsecond=0)
        
        now_local = now_utc.astimezone()
        rounded_local = now_local.replace(minute=0, second=0, microsecond=0)
        
        return (
            "Runtime Clock (Rounded to Hour):\n"
            f"- Current UTC time: {rounded_utc.isoformat()}\n"
            f"- Current server local time: {rounded_local.isoformat()}\n"
            "Treat this clock as a general reference for today's date and hour. "
            "For sub-minute trade timing, rely on live tool-call results."
        )

    def _build_json_config(self, schema: Optional[Any] = None, system_instruction: Optional[str] = None) -> types.GenerateContentConfig:
        system_instruction = self._apply_language_instruction(system_instruction)
        logger.info("[LLMClient] Building JSON config with runtime clock injection.")
        return types.GenerateContentConfig(
            temperature=settings.VERTEX_AI_TEMPERATURE,
            max_output_tokens=settings.VERTEX_AI_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_json_schema=schema,
            system_instruction=system_instruction,
            # Controlled generation (JSON mode) is incompatible with Search tool on Vertex
            tools=self._build_vertex_server_tools(exclude_server_tools=True),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=15
            )
        )

    def _build_text_config(self, system_instruction: Optional[str] = None) -> types.GenerateContentConfig:
        system_instruction = self._apply_language_instruction(system_instruction)
        logger.info("[LLMClient] Building text config with runtime clock injection.")
        return types.GenerateContentConfig(
            temperature=settings.VERTEX_AI_TEMPERATURE,
            max_output_tokens=settings.VERTEX_AI_MAX_OUTPUT_TOKENS,
            system_instruction=system_instruction,
            tools=self._build_vertex_server_tools(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=15
            )
        )

    def _build_vertex_server_tools(
        self, exclude_server_tools: bool = False
    ) -> List[Any]:
        """
        Optional Vertex-managed server tools.
        Returns a list containing at most one tool to avoid API conflicts.
        """
        if exclude_server_tools:
            return []

        # Priority selection for Google-managed server tools
        if settings.VERTEX_AI_ENABLE_GOOGLE_SEARCH:
            # Vertex AI REQUIRES the 'google_search' key, NOT 'google_search_retrieval'
            return [{"google_search": {}}]
        
        if settings.VERTEX_AI_ENABLE_URL_CONTEXT:
            return [{"url_context": {}}]
            
        if settings.VERTEX_AI_ENABLE_CODE_EXECUTION:
            return [types.Tool(code_execution=types.CodeExecution())]
            
        return []

    def _get_cache_resource_key(self, model: str, system_instruction: str) -> str:
        """
        Create a stable key for identifying a context cache.
        """
        payload = f"{model}:{system_instruction}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _get_or_create_context_cache(
        self, 
        model: str, 
        system_instruction: str, 
        trace_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Manage the lifecycle of a Vertex AI Context Cache.
        Returns the full resource name of the cache if successful.
        """
        if not settings.VERTEX_AI_ENABLE_CONTEXT_CACHING:
            return None

        client = self._ensure_client()
        if client is None:
            return None

        cache_key = self._get_cache_resource_key(model, system_instruction)
        
        # Check in-memory first
        existing_cache = self._context_caches.get(cache_key)
        if existing_cache:
            # Basic validation: Check if it still exists on the server? 
            # For brevity and speed, we trust the in-memory cache until it fails.
            return existing_cache

        try:
            started = time.monotonic()
            
            # Verify token count meets minimum threshold (1024 tokens)
            token_resp = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: client.models.count_tokens(
                    model=model,
                    contents=system_instruction
                )
            )
            token_count = token_resp.total_tokens
            if token_count < 1024:
                logger.info(
                    "[LLMClient][%s] Skipping Context Cache | token_count=%s (min=1024) | model=%s",
                    trace_id or "no-trace",
                    token_count,
                    model
                )
                return None

            logger.info(
                "[LLMClient][%s] Creating new Context Cache | model=%s | tokens=%s",
                trace_id or "no-trace",
                model,
                token_count
            )
            
            # Note: Context Caching requires the system instruction to be part of the cache
            # The SDK expects types.CreateCachedContentConfig
            cached_content = await asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: client.caches.create(
                    model=model,
                    config=types.CreateCachedContentConfig(
                        system_instruction=system_instruction,
                        display_name=f"aidaan-cache-{cache_key[:8]}",
                        ttl=f"{settings.VERTEX_AI_CONTEXT_CACHE_TTL_SECONDS}s",
                    ),
                )
            )
            
            cache_name = cached_content.name
            self._context_caches[cache_key] = cache_name
            
            logger.info(
                "[LLMClient][%s] Context Cache created | name=%s | latency_ms=%s",
                trace_id or "no-trace",
                cache_name,
                self._elapsed_ms(started)
            )
            return cache_name
        except Exception as exc:
            # Log but don't fail; falling back to non-cached generation is safer
            logger.warning(
                "[LLMClient][%s] Failed to create Context Cache: %s",
                trace_id or "no-trace",
                exc
            )
            return None

    def _clean_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reduce MCP JSON schema to a function-calling safe subset.
        """
        cleaned = dict(schema)

        if "anyOf" in cleaned:
            non_null = [item for item in cleaned["anyOf"] if item.get("type") != "null"]
            if non_null:
                cleaned.update(self._clean_schema(non_null[0]))
            cleaned.pop("anyOf", None)

        if "oneOf" in cleaned:
            non_null = [item for item in cleaned["oneOf"] if item.get("type") != "null"]
            if non_null:
                cleaned.update(self._clean_schema(non_null[0]))
            cleaned.pop("oneOf", None)

        for field in [
            "title",
            "additionalProperties",
            "default",
            "examples",
            "$schema",
        ]:
            cleaned.pop(field, None)

        properties = cleaned.get("properties")
        if isinstance(properties, dict):
            cleaned["properties"] = {
                key: self._clean_schema(value)
                for key, value in properties.items()
                if isinstance(value, dict)
            }
            if "type" not in cleaned:
                cleaned["type"] = "object"

        items = cleaned.get("items")
        if isinstance(items, dict):
            cleaned["items"] = self._clean_schema(items)

        return cleaned

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        if not raw_text or not raw_text.strip():
            raise ValueError("Empty or whitespace text provided for JSON extraction")

        clean = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        payload = match.group(0) if match else clean
        
        try:
            # Phase 4 Fix: Ensure proper Unicode handling for Hindi/Gujarati/Marathi text
            # Use ensure_ascii=False during dumps to preserve Unicode characters
            parsed = json.loads(payload)
            
            # Validate and sanitize the parsed JSON to ensure it's serializable
            # This prevents Unicode escape issues in downstream processing
            validated = json.loads(json.dumps(parsed, ensure_ascii=False))
            return validated
        except json.JSONDecodeError as exc:
            logger.error("[LLMClient] JSON parsing failed for payload: %s", payload[:200])
            raise exc

    @staticmethod
    def _is_valid_unicode_escapes(text: str) -> bool:
        r"""
        Check if all Unicode escape sequences in the text are valid.
        Returns False if any invalid \uXXXX sequences are found.
        """
        # Find all \uXXXX patterns
        unicode_pattern = r'\\u[0-9a-fA-F]{0,4}'
        matches = re.findall(unicode_pattern, text)
        
        for match in matches:
            # Valid Unicode escape must be exactly \uXXXX where X is hex digit
            if len(match) != 6:  # \u + 4 hex digits
                return False
        return True

    @staticmethod
    def _repair_unicode_escapes(text: str) -> str:
        """
        Attempt to repair malformed Unicode escape sequences.
        Converts raw Unicode characters to proper JSON-safe format.
        """
        try:
            # Strategy 1: Try to decode any valid escapes and re-encode properly
            # This handles cases where the model mixed escaped and unescaped Unicode
            decoded = text.encode('utf-8').decode('unicode-escape', errors='ignore')
            # Re-encode to ensure proper JSON format
            repaired = json.dumps(decoded, ensure_ascii=False)[1:-1]  # Remove quotes
            return repaired
        except Exception:
            # Strategy 2: If that fails, just remove invalid escape sequences
            # Replace incomplete \uXXX with the literal characters
            repaired = re.sub(r'\\u([0-9a-fA-F]{0,3})(?![0-9a-fA-F])', r'\1', text)
            return repaired

    async def generate_json(
        self,
        prompt: str,
        model_override: Optional[str] = None,
        retries: int = 3,
        use_mcp_tools: bool = False,
        tool_callback: Optional[Callable[[str], Any]] = None,
        response_schema: Optional[Any] = None,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON using Vertex AI.
        Uses Hybrid Mode (Text + JSON Instructions) when tools are enabled to bypass Vertex API conflicts.
        """
        model_name = model_override or self._default_model_name
        cache_prefix = "json_tools" if use_mcp_tools else "json"
        trace_id = self._new_trace_id("json")
        overall_start = time.monotonic()
        
        # Determine if we need to bypass strict JSON mode due to tool availability
        # On Vertex AI, Search tools are incompatible with response_mime_type="application/json"
        has_server_tools = bool(self._build_vertex_server_tools())
        use_hybrid_mode = use_mcp_tools or has_server_tools

        cache_key = self._cache_key(
            cache_prefix,
            model_name,
            prompt,
            schema=response_schema,
            system_instruction=system_instruction,
            use_mcp_tools=use_mcp_tools,
        )
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            self._log_usage("generate_json", prompt, cached=True)
            logger.info(
                "[LLMClient][%s] Cache hit | model=%s | use_mcp_tools=%s | total_ms=%s",
                trace_id,
                model_name,
                use_mcp_tools,
                self._elapsed_ms(overall_start),
            )
            return cached_result

        # REDACT PII via DLP before Vertex processes it
        dlp_start = time.monotonic()
        logger.info(f"[TIMING] Starting DLP redaction | trace_id={trace_id}")
        prompt = await asyncio.get_event_loop().run_in_executor(_executor, dlp_client.redact_pii, prompt)
        dlp_elapsed = self._elapsed_ms(dlp_start)
        logger.info("[LLMClient][%s] DLP redaction completed | latency_ms=%s", trace_id, dlp_elapsed)
        logger.info(f"[TIMING] DLP completed | elapsed={dlp_elapsed/1000:.3f}s")

        # Helper: add JSON instruction for non-strict calls
        if use_hybrid_mode and "Return ONLY raw JSON" not in prompt:
            prompt += "\n\nCRITICAL: Return ONLY raw JSON. No conversational filler or markdown blocks."

        if use_mcp_tools:
            result = await self._generate_json_with_mcp_tools(
                prompt=prompt,
                model_name=model_name,
                retries=retries,
                tool_callback=tool_callback,
                response_schema=response_schema,
                system_instruction=system_instruction,
                trace_id=trace_id,
            )
            cache.set(cache_key, result, expire=settings.LLM_CACHE_EXPIRE)
            self._log_usage("generate_json", prompt)
            logger.info(
                "[LLMClient][%s] generate_json completed | model=%s | path=mcp_tools | total_ms=%s",
                trace_id,
                model_name,
                self._elapsed_ms(overall_start),
            )
            return result

        start = time.monotonic()
        last_error: Optional[Exception] = None
        client = self._ensure_client()
        if client is None:
            return self._vertex_error_payload()

        # Choose config: Hybrid/Text config if search is enabled, else strict JSON config
        config = self._build_text_config(system_instruction=system_instruction) if use_hybrid_mode else self._build_json_config(schema=response_schema, system_instruction=system_instruction)

        # Apply Context Caching if enabled and compatible
        # Note: We apply this AFTER building the config because building the config 
        # applies the Multi-Language and Timestamp instructions which we want cached.
        if settings.VERTEX_AI_ENABLE_CONTEXT_CACHING and config.system_instruction:
            cache_name = await self._get_or_create_context_cache(
                model=model_name,
                system_instruction=config.system_instruction,
                trace_id=trace_id
            )
            if cache_name:
                config.cached_content = cache_name
                # IMPORTANT: When using a cache, the system_instruction must be cleared 
                # from the request config to avoid redundancy/errors.
                config.system_instruction = None

        for attempt in range(retries):
            try:
                response = await self._run_generate_content(
                    client=client,
                    model=model_name,
                    contents=prompt,
                    config=config,
                    trace_id=trace_id,
                    stage=f"generate_json.attempt_{attempt + 1}",
                )
                raw = (response.text or "").strip()
                
                # Phase 4 Fix: Validate response before parsing to catch encoding issues early
                if not raw:
                    raise ValueError("Model returned empty response")
                
                # Check for common Unicode escape issues in the raw response
                if r"\u" in raw and not self._is_valid_unicode_escapes(raw):
                    logger.warning("[LLMClient][%s] Detected invalid Unicode escapes in response, attempting repair", trace_id)
                    raw = self._repair_unicode_escapes(raw)
                
                parsed = self._extract_json(raw)
                # NOTE: DLP redaction on LLM-generated responses is intentionally skipped.
                # The model does not reproduce raw user PII in structured JSON output.
                # Input-side redaction (above) is sufficient and avoids double latency.

                cache.set(cache_key, parsed, expire=settings.LLM_CACHE_EXPIRE)
                self._log_usage("generate_json", prompt)
                latency_ms = int((time.monotonic() - start) * 1000)
                logger.debug(
                    "[LLMClient] generate_json OK | mode=%s | model=%s | latency=%sms",
                    "hybrid" if use_hybrid_mode else "strict",
                    model_name,
                    latency_ms,
                )
                audit_logger.log_llm_transaction(
                    prompt=prompt,
                    response=parsed,
                    model_id=model_name,
                    latency_ms=latency_ms,
                    status="SUCCESS"
                )
                return parsed
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[LLMClient][%s] generate_json failed (attempt %s/%s): %s",
                    trace_id,
                    attempt + 1,
                    retries,
                    exc,
                )
                if attempt < retries - 1:
                    is_429 = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                    # Exponential Backoff with Jitter and 429-specific padding
                    base_delay = 2 ** attempt
                    jitter = (time.time() % 1)
                    delay = (base_delay * 3 + 2 + jitter) if is_429 else (base_delay + jitter)
                    
                    logger.info("[LLMClient][%s] Retrying generate_json in %0.2fs due to: %s", trace_id, delay, exc)
                    await asyncio.sleep(delay)
                else:
                    logger.error("[LLMClient][%s] generate_json exhausted retries: %s", trace_id, last_error)
                    audit_logger.log_llm_transaction(
                        prompt=prompt,
                        response=None,
                        model_id=model_name,
                        latency_ms=int((time.monotonic() - start) * 1000),
                        status="ERROR"
                    )
                    return {"error": str(last_error) if last_error else "Vertex AI JSON generation failed"}

    async def _generate_json_with_mcp_tools(
        self,
        prompt: str,
        model_name: str,
        retries: int,
        tool_callback: Optional[Callable[[str], Any]] = None,
        response_schema: Optional[Any] = None,
        system_instruction: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        client = self._ensure_client()
        if client is None:
            return self._vertex_error_payload()
        last_error: Optional[Exception] = None
        start = time.monotonic()

        async with self._mcp_session_context(trace_id=trace_id) as session:
            if not session:
                return {"error": "MCP session not available"}

            mcp_tools = await self._list_mcp_tools(session, trace_id=trace_id)
            if not mcp_tools:
                return {"error": "No MCP tools available"}

            declarations = []
            for tool in mcp_tools:
                schema = self._clean_schema(tool.inputSchema.copy())
                declarations.append(
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description or f"MCP tool: {tool.name}",
                        parameters_json_schema=schema,
                    )
                )

            tool_config = types.GenerateContentConfig(
                temperature=settings.VERTEX_AI_TEMPERATURE,
                max_output_tokens=settings.VERTEX_AI_MAX_OUTPUT_TOKENS,
                tools=[types.Tool(function_declarations=declarations)],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=False,
                    maximum_remote_calls=15
                )
            )

            user_prompt_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
            contents = [user_prompt_content]
            try:
                tool_config.system_instruction = system_instruction
                seen_tool_signatures: set[str] = set()

                response = await self._run_generate_content(
                    client=client,
                    model=model_name,
                    contents=contents,
                    config=tool_config,
                    trace_id=trace_id,
                    stage="mcp.initial_turn",
                )

                max_turns = 15
                turn = 0
                while response.function_calls and turn < max_turns:
                    turn += 1
                    
                    batch_start = time.monotonic()
                    function_call_content = response.candidates[0].content
                    contents.append(function_call_content)

                    tasks = []
                    queued_calls = []
                    tools_in_batch = 0
                    max_tools_per_turn = settings.VERTEX_AI_MAX_TOOLS_PER_TURN
                    
                    for function_call in response.function_calls:
                        tool_name = function_call.name
                        tool_args = dict(function_call.args or {})
                        signature = self._tool_call_signature(tool_name, tool_args)
                        if signature in seen_tool_signatures:
                            logger.info(
                                "[LLMClient][%s] [%s/%s] Skipping duplicate tool call: %s",
                                trace_id or "no-trace",
                                turn,
                                max_turns,
                                tool_name,
                            )
                            continue
                        
                        # Limit tools per turn to prevent context overflow
                        if tools_in_batch >= max_tools_per_turn:
                            logger.warning(
                                "[LLMClient][%s] [%s/%s] Tool limit reached (%s/%s), skipping remaining tools",
                                trace_id or "no-trace",
                                turn,
                                max_turns,
                                tools_in_batch,
                                max_tools_per_turn,
                            )
                            break
                        
                        seen_tool_signatures.add(signature)
                        tools_in_batch += 1
                        logger.info(
                            "[LLMClient][%s] [%s/%s] Executing tool (Parallel): %s",
                            trace_id or "no-trace",
                            turn,
                            max_turns,
                            tool_name,
                        )
                        if tool_callback:
                            await tool_callback(tool_name)
                        queued_calls.append(function_call)
                        tasks.append(
                            self._execute_mcp_tool(
                                session,
                                tool_name,
                                tool_args,
                                trace_id=trace_id,
                            )
                        )

                    if not tasks:
                        logger.info(
                            "[LLMClient][%s] No unique tool calls left in turn=%s; breaking MCP loop.",
                            trace_id or "no-trace",
                            turn,
                        )
                        break

                    results = await asyncio.gather(*tasks)
                    batch_elapsed = self._elapsed_ms(batch_start)
                    logger.info(
                        "[LLMClient][%s] Tool batch completed | turn=%s | tool_calls=%s | latency_ms=%s",
                        trace_id or "no-trace",
                        turn,
                        len(tasks),
                        batch_elapsed,
                    )
                    logger.info(f"[TIMING] Tool batch {turn} completed | tools={len(tasks)} | elapsed={batch_elapsed/1000:.3f}s")

                    tool_parts = []
                    for function_call, tool_result in zip(queued_calls, results):
                        tool_parts.append(
                            types.Part.from_function_response(
                                name=function_call.name,
                                response={"result": tool_result},
                            )
                        )

                    contents.append(types.Content(role="tool", parts=tool_parts))
                    tool_config.system_instruction = system_instruction

                    response = await self._run_generate_content(
                        client=client,
                        model=model_name,
                        contents=contents,
                        config=tool_config,
                        trace_id=trace_id,
                        stage=f"mcp.followup_turn_{turn}",
                    )

                if not response.function_calls and response_schema:
                    config = self._build_json_config(
                        schema=response_schema,
                        system_instruction=system_instruction,
                    )
                    # Speed Optimization: Use the faster default model (Flash) for final synthesis
                    # after the reasoning model (Pro) has completed its tool calls and analysis.
                    synthesis_model = self._default_model_name
                    logger.info(
                        "[LLMClient][%s] Switching to synthesis model: %s",
                        trace_id,
                        synthesis_model
                    )
                    response = await self._run_generate_content(
                        client=client,
                        model=synthesis_model,
                        contents=contents,
                        config=config,
                        trace_id=trace_id,
                        stage="mcp.final_synthesis",
                    )

                raw = (response.text or "").strip()
                if not raw:
                    raise ValueError("Model returned no text after MCP tool usage")

                # Phase 4 Fix: Validate response before parsing
                if r"\u" in raw and not self._is_valid_unicode_escapes(raw):
                    logger.warning("[LLMClient][%s] Detected invalid Unicode escapes in MCP response, attempting repair", trace_id)
                    raw = self._repair_unicode_escapes(raw)

                parsed = self._extract_json(raw)
                # NOTE: DLP redaction on LLM-generated responses is intentionally skipped.
                # Input-side redaction is sufficient; model output does not reproduce raw PII.

                latency_ms = int((time.monotonic() - start) * 1000)
                logger.debug(
                    "[LLMClient] generate_json_with_mcp_tools OK | model=%s | latency=%sms",
                    model_name,
                    latency_ms,
                )
                audit_logger.log_llm_transaction(
                    prompt=prompt,
                    response=parsed,
                    model_id=model_name,
                    latency_ms=latency_ms,
                    status="SUCCESS"
                )
                return parsed
            except Exception as exc:
                last_error = exc
                logger.error(
                    "[LLMClient][%s] MCP tool loop failed | elapsed_ms=%s: %s",
                    trace_id or "no-trace",
                    self._elapsed_ms(start),
                    exc,
                )

        return {"error": str(last_error) if last_error else "Vertex AI tool loop failed"}

    async def generate_text(
        self,
        prompt: str,
        model_override: Optional[str] = None,
    ) -> str:
        """
        Send a prompt and return raw text response.
        """
        model_name = model_override or self._default_model_name
 
        # REDACT PII via DLP before Vertex processes it
        prompt = await asyncio.get_event_loop().run_in_executor(_executor, dlp_client.redact_pii, prompt)

        cache_key = self._cache_key("text", model_name, prompt)
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            self._log_usage("generate_text", prompt, cached=True)
            return cached_result
        client = self._ensure_client()
        if client is None:
            logger.error("[LLMClient] generate_text skipped: %s", self._vertex_disabled_reason)
            return ""

        try:
            config = self._build_text_config()
            
            # Apply Context Caching for text generation
            if settings.VERTEX_AI_ENABLE_CONTEXT_CACHING and config.system_instruction:
                cache_name = await self._get_or_create_context_cache(
                    model=model_name,
                    system_instruction=config.system_instruction,
                    trace_id="text-gen"
                )
                if cache_name:
                    config.cached_content = cache_name
                    config.system_instruction = None

            response = await self._run_generate_content(
                client=client,
                model=model_name,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            # NOTE: DLP redaction on LLM-generated text responses is intentionally skipped.
            # Input-side redaction is sufficient; model output does not reproduce raw PII.

            cache.set(cache_key, text, expire=settings.LLM_CACHE_EXPIRE)
            self._log_usage("generate_text", prompt)
            audit_logger.log_llm_transaction(
                prompt=prompt,
                response=text,
                model_id=model_name,
                latency_ms=0,
                status="SUCCESS"
            )
            return text
        except Exception as exc:
            logger.error("[LLMClient] generate_text failed: %s: %s", type(exc).__name__, exc)
            audit_logger.log_llm_transaction(
                prompt=prompt,
                response=None,
                model_id=model_name,
                latency_ms=0,
                status="ERROR"
            )
            return ""

    def generate_json_sync(
        self,
        prompt: str,
        model_override: Optional[str] = None,
        timeout: int = 8,
    ) -> Dict[str, Any]:
        """
        Synchronous blocking version used in threadpool contexts.
        """
        model_name = model_override or self._default_model_name

        # REDACT PII via DLP before Vertex processes it
        prompt = dlp_client.redact_pii(prompt)

        cache_key = self._cache_key("json_sync", model_name, prompt)
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            self._log_usage("generate_json_sync", prompt, cached=True)
            return cached_result

        start = time.monotonic()
        client = self._ensure_client()
        if client is None:
            return self._vertex_error_payload()

        try:
            future = _executor.submit(
                lambda: client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=self._build_json_config(),
                )
            )
            result_raw = future.result(timeout=timeout)
            raw = (result_raw.text or "").strip()
            parsed = self._extract_json(raw)
            cache.set(cache_key, parsed, expire=settings.LLM_CACHE_EXPIRE)
            self._log_usage("generate_json_sync", prompt)

            latency_ms = int((time.monotonic() - start) * 1000)
            logger.debug(
                "[LLMClient] generate_json_sync OK | model=%s | latency=%sms",
                model_name,
                latency_ms,
            )
            audit_logger.log_llm_transaction(
                prompt=prompt,
                response=parsed,
                model_id=model_name,
                latency_ms=latency_ms,
                status="SUCCESS"
            )
            return parsed
        except json.JSONDecodeError as exc:
            logger.warning("[LLMClient] sync JSON parse error: %s", exc)
            audit_logger.log_llm_transaction(
                prompt=prompt,
                response=None,
                model_id=model_name,
                latency_ms=int((time.monotonic() - start) * 1000),
                status="ERROR"
            )
            return {"error": "JSON parse failed"}
        except Exception as exc:
            logger.error("[LLMClient] sync call failed: %s: %s", type(exc).__name__, exc)
            audit_logger.log_llm_transaction(
                prompt=prompt,
                response=None,
                model_id=model_name,
                latency_ms=int((time.monotonic() - start) * 1000),
                status="ERROR"
            )
            return {"error": str(exc)}


llm_client = LLMClient()
