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
import hashlib
import json
import logging
import os
import re
import shlex
import sys
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from app.core.cache import cache
from app.core.config.settings import settings

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vertex_llm_worker")


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
        self._mcp_session = None
        self._mcp_exit_stack = None
        self._mcp_inprocess_server = None

        self._configure_vertex_auth()
        self._ensure_client()

    def _configure_vertex_auth(self) -> None:
        """
        Configure service-account based auth when an explicit path is supplied.
        Otherwise Vertex AI falls back to Application Default Credentials.
        """
        if settings.VERTEX_AI_SERVICE_ACCOUNT_JSON:
            try:
                credential_dir = Path(tempfile.gettempdir())
                credential_dir.mkdir(parents=True, exist_ok=True)
                credential_path = credential_dir / "aidaan-vertex-service-account.json"
                credential_path.write_text(settings.VERTEX_AI_SERVICE_ACCOUNT_JSON, encoding="utf-8")
                self._temp_credential_file = credential_path
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)
                logger.info(
                    "[LLMClient] Using service account credentials from VERTEX_AI_SERVICE_ACCOUNT_JSON"
                )
                return
            except Exception as exc:
                logger.error("[LLMClient] Failed to persist VERTEX_AI_SERVICE_ACCOUNT_JSON: %s", exc)

        credential_path = settings.VERTEX_AI_SERVICE_ACCOUNT_FILE
        if not credential_path:
            return

        resolved_path = Path(credential_path).expanduser()
        if not resolved_path.is_file():
            logger.warning(
                "[LLMClient] VERTEX_AI_SERVICE_ACCOUNT_FILE does not exist: %s",
                resolved_path,
            )
            return

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved_path)
        logger.info("[LLMClient] Using service account credentials from %s", resolved_path)

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

    async def _run_generate_content(
        self,
        client: genai.Client,
        model: str,
        contents: Any,
        config: types.GenerateContentConfig,
    ) -> Any:
        return await asyncio.get_event_loop().run_in_executor(
            _executor,
            lambda: client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            ),
        )

    async def _ensure_mcp_session(self):
        """
        Connect to the MCP server and maintain a long-lived session.
        """
        if settings.MCP_TRANSPORT == "inprocess":
            if self._mcp_inprocess_server is not None:
                return self._mcp_inprocess_server
            try:
                from app.services.aidaan.mcp.market import mcp as market_mcp

                self._mcp_inprocess_server = market_mcp
                logger.info("[LLMClient] Using in-process MCP server.")
                return self._mcp_inprocess_server
            except Exception as exc:
                logger.error("[LLMClient] In-process MCP server import failed: %s", exc)
                return None

        if self._mcp_session:
            return self._mcp_session

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from contextlib import AsyncExitStack

            self._mcp_exit_stack = AsyncExitStack()
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

            # Initialize stdio transport and session
            read, write = await self._mcp_exit_stack.enter_async_context(stdio_client(server_params))
            session = await self._mcp_exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            
            self._mcp_session = session
            logger.info("[LLMClient] Persistent MCP session initialized.")
            return self._mcp_session
        except Exception as exc:
            logger.error("[LLMClient] Failed to initialize persistent MCP session: %s", exc)
            if self._mcp_exit_stack is not None:
                try:
                    await self._mcp_exit_stack.aclose()
                except Exception:
                    pass
            self._mcp_exit_stack = None
            self._mcp_session = None
            return None

    async def _ensure_mcp_tools(self):
        """
        Fetch tools via the persistent session.
        """
        if self._mcp_tools:
            return self._mcp_tools

        session = await self._ensure_mcp_session()
        if not session:
            return []

        try:
            if settings.MCP_TRANSPORT == "inprocess":
                self._mcp_tools = await session.list_tools()
            else:
                tools_resp = await session.list_tools()
                self._mcp_tools = tools_resp.tools
            logger.info("[LLMClient] Discovered %s tools via MCP (%s).", len(self._mcp_tools), settings.MCP_TRANSPORT)
            return self._mcp_tools
        except Exception as exc:
            logger.error("[LLMClient] MCP tool discovery failed: %s", exc)
            return []

    async def _execute_mcp_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a specific tool via the long-lived MCP session.
        """
        session = await self._ensure_mcp_session()
        if not session:
            return {"error": "MCP session not available."}

        try:
            if settings.MCP_TRANSPORT == "inprocess":
                content, meta = await session.call_tool(name, arguments)
                if isinstance(meta, dict) and isinstance(meta.get("result"), dict):
                    return meta["result"]
                return self._normalize_mcp_result(content)

            result = await session.call_tool(name, arguments)
            return self._normalize_mcp_result(result.content)
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
                normalized.append(json.loads(text))
            except Exception:
                normalized.append(text)

        return normalized[0] if len(normalized) == 1 else normalized

    async def call_mcp_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Public wrapper for explicit MCP tool execution from agents/services.
        """
        return await self._execute_mcp_tool(name, arguments)

    def _cache_key(self, prefix: str, model_name: str, payload: str) -> str:
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
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

    def _build_json_config(self, schema: Optional[Any] = None, system_instruction: Optional[str] = None) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=settings.VERTEX_AI_TEMPERATURE,
            max_output_tokens=settings.VERTEX_AI_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_json_schema=schema,
            system_instruction=system_instruction,
            # Controlled generation (JSON mode) is incompatible with Search tool on Vertex
            tools=self._build_vertex_server_tools(exclude_server_tools=True),
        )

    def _build_text_config(self, system_instruction: Optional[str] = None) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=settings.VERTEX_AI_TEMPERATURE,
            max_output_tokens=settings.VERTEX_AI_MAX_OUTPUT_TOKENS,
            system_instruction=system_instruction,
            tools=self._build_vertex_server_tools(),
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
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.error("[LLMClient] JSON parsing failed for payload: %s", payload[:200])
            raise exc

    async def generate_json(
        self,
        prompt: str,
        model_override: Optional[str] = None,
        retries: int = 2,
        use_mcp_tools: bool = False,
        tool_callback: Optional[callable] = None,
        response_schema: Optional[Any] = None,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON using Vertex AI.
        Uses Hybrid Mode (Text + JSON Instructions) when tools are enabled to bypass Vertex API conflicts.
        """
        model_name = model_override or self._default_model_name
        cache_prefix = "json_tools" if use_mcp_tools else "json"
        
        # Determine if we need to bypass strict JSON mode due to tool availability
        # On Vertex AI, Search tools are incompatible with response_mime_type="application/json"
        has_server_tools = bool(self._build_vertex_server_tools())
        use_hybrid_mode = use_mcp_tools or has_server_tools

        cache_key = self._cache_key(cache_prefix, model_name, prompt)
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            self._log_usage("generate_json", prompt, cached=True)
            return cached_result

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
            )
            cache.set(cache_key, result, expire=settings.LLM_CACHE_EXPIRE)
            self._log_usage("generate_json", prompt)
            return result

        start = time.monotonic()
        last_error: Optional[Exception] = None
        client = self._ensure_client()
        if client is None:
            return self._vertex_error_payload()

        # Choose config: Hybrid/Text config if search is enabled, else strict JSON config
        config = self._build_text_config(system_instruction=system_instruction) if use_hybrid_mode else self._build_json_config(schema=response_schema, system_instruction=system_instruction)

        for attempt in range(retries):
            try:
                response = await self._run_generate_content(
                    client=client,
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                raw = (response.text or "").strip()
                parsed = self._extract_json(raw)
                cache.set(cache_key, parsed, expire=settings.LLM_CACHE_EXPIRE)
                self._log_usage("generate_json", prompt)
                latency_ms = int((time.monotonic() - start) * 1000)
                logger.debug(
                    "[LLMClient] generate_json OK | mode=%s | model=%s | latency=%sms",
                    "hybrid" if use_hybrid_mode else "strict",
                    model_name,
                    latency_ms,
                )
                return parsed
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[LLMClient] generate_json failed (attempt %s/%s): %s",
                    attempt + 1,
                    retries,
                    exc,
                )
                if attempt < retries - 1:
                    await asyncio.sleep(1)

        logger.error("[LLMClient] generate_json exhausted retries: %s", last_error)
        return {"error": str(last_error) if last_error else "Vertex AI JSON generation failed"}

    async def _generate_json_with_mcp_tools(
        self,
        prompt: str,
        model_name: str,
        retries: int,
        tool_callback: Optional[callable] = None,
        response_schema: Optional[Any] = None,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        mcp_tools = await self._ensure_mcp_tools()
        if not mcp_tools:
            return {"error": "No MCP tools available"}
        client = self._ensure_client()
        if client is None:
            return self._vertex_error_payload()

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

        # PRIORITY FIX: We avoid mixing built-in server tools (search, etc.) with 
        # custom function declarations, as the API often rejects multiple tool types 
        # unless they are all search tools.
        tool_config = types.GenerateContentConfig(
            temperature=settings.VERTEX_AI_TEMPERATURE,
            max_output_tokens=settings.VERTEX_AI_MAX_OUTPUT_TOKENS,
            tools=[types.Tool(function_declarations=declarations)],
        )

        user_prompt_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
        last_error: Optional[Exception] = None
        start = time.monotonic()

        for attempt in range(retries):
            contents = [user_prompt_content]
            try:
                # Update tool_config with system_instruction before initial turn
                tool_config.system_instruction = system_instruction

                response = await self._run_generate_content(
                    client=client,
                    model=model_name,
                    contents=contents,
                    config=tool_config,
                )

                max_turns = 5
                turn = 0
                while response.function_calls and turn < max_turns:
                    turn += 1
                    function_call_content = response.candidates[0].content
                    contents.append(function_call_content)

                    # PARALLEL EXECUTION: Fire all tool calls suggested in this turn at once.
                    tasks = []
                    for function_call in response.function_calls:
                        tool_name = function_call.name
                        tool_args = dict(function_call.args or {})
                        logger.info(
                            "[LLMClient] [%s/%s] Executing tool (Parallel): %s",
                            turn,
                            max_turns,
                            tool_name,
                        )
                        if tool_callback:
                            await tool_callback(tool_name)
                        tasks.append(self._execute_mcp_tool(tool_name, tool_args))

                    results = await asyncio.gather(*tasks)

                    tool_parts = []
                    for function_call, tool_result in zip(response.function_calls, results):
                        tool_parts.append(
                            types.Part.from_function_response(
                                name=function_call.name,
                                response={"result": tool_result},
                            )
                        )

                    contents.append(types.Content(role="tool", parts=tool_parts))

                    # Handle system_instruction within tool_config if needed
                    tool_config.system_instruction = system_instruction

                    response = await self._run_generate_content(
                        client=client,
                        model=model_name,
                        contents=contents,
                        config=tool_config,
                    )

                # Final Synthesis Turn override logic:
                if not response.function_calls and response_schema:
                    config = self._build_json_config(schema=response_schema, system_instruction=system_instruction)
                    response = await self._run_generate_content(
                        client=client,
                        model=model_name,
                        contents=contents,
                        config=config,
                    )

                raw = (response.text or "").strip()
                if not raw:
                    raise ValueError("Model returned no text after MCP tool usage")

                parsed = self._extract_json(raw)
                latency_ms = int((time.monotonic() - start) * 1000)
                logger.debug(
                    "[LLMClient] generate_json_with_mcp_tools OK | model=%s | latency=%sms",
                    model_name,
                    latency_ms,
                )
                return parsed
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[LLMClient] MCP tool loop failed (attempt %s/%s): %s",
                    attempt + 1,
                    retries,
                    exc,
                )
                if attempt < retries - 1:
                    await asyncio.sleep(1)

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
            response = await self._run_generate_content(
                client=client,
                model=model_name,
                contents=prompt,
                config=self._build_text_config(),
            )
            text = (response.text or "").strip()
            cache.set(cache_key, text, expire=settings.LLM_CACHE_EXPIRE)
            self._log_usage("generate_text", prompt)
            return text
        except Exception as exc:
            logger.error("[LLMClient] generate_text failed: %s: %s", type(exc).__name__, exc)
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
            return parsed
        except json.JSONDecodeError as exc:
            logger.warning("[LLMClient] sync JSON parse error: %s", exc)
            return {"error": "JSON parse failed"}
        except Exception as exc:
            logger.error("[LLMClient] sync call failed: %s: %s", type(exc).__name__, exc)
            return {"error": str(exc)}


llm_client = LLMClient()
