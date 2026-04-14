"""
AIDAAN Generic LLM Client
==========================
One place to configure, call, and debug ALL Gemini interactions.

HOW TO USE:
    from app.core.llm_client import llm_client

    # Simple JSON call (returns parsed dict)
    result = await llm_client.generate_json(prompt)

    # Raw text call
    text = await llm_client.generate_text(prompt)

HOW TO DEBUG:
    - Set LOG_LEVEL=DEBUG in your .env to see every prompt + raw response
    - Every call logs: [LLM] model | tokens_approx | latency_ms
    - Failures are logged with full prompt (truncated to 500 chars)

HOW TO SWAP MODELS:
    - Change VERTEX_AI_MODEL_NAME in settings.py — affects all agents instantly
    - Or pass model_override="gemini-1.0-pro" to generate_json() for one-off calls
"""
import json
import logging
import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.core.config.settings import settings

logger = logging.getLogger(__name__)

# One shared thread pool for all blocking Gemini SDK calls
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm_worker")


class LLMClient:
    """
    Generic async wrapper around the Gemini SDK.
    Single instance used by all agents via dependency injection or singleton.
    """

    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self._default_model_name = settings.VERTEX_AI_MODEL_NAME
        self._model_cache: Dict[str, genai.GenerativeModel] = {}
        self._mcp_tools = []
        self._mcp_client = None

        logger.info(f"[LLMClient] Initialized. Default model: {self._default_model_name}")

        # Startup smoke test - Disabled to save quota
        # self._smoke_test()

    # ------------------------------------------------------------------
    def _get_model(
        self,
        model_name: Optional[str] = None
    ) -> genai.GenerativeModel:
        """
        Returns a cached GenerativeModel instance.
        Creates one if it doesn't exist yet.
        """
        name = model_name or self._default_model_name
        if name not in self._model_cache:
            self._model_cache[name] = genai.GenerativeModel(
                model_name=name,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                }
            )
            logger.debug(f"[LLMClient] Created model instance: {name}")
        return self._model_cache[name]

    def _smoke_test(self):
        """
        Quick startup check. Logs OK or FAIL — never raises.
        """
        try:
            model = self._get_model()
            res = model.generate_content('Return JSON: {"status": "ok"}')
            logger.info(f"[LLMClient] Smoke test OK: {res.text[:60]}")
        except Exception as e:
            logger.error(f"[LLMClient] Smoke test FAILED: {e}")

    async def _ensure_mcp_tools(self):
        """
        Connects to the MCP server, fetches tools, and caches them.
        """
        if self._mcp_tools:
            return self._mcp_tools

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=settings.MCP_SERVER_COMMAND,
                args=[settings.MCP_SERVER_ARGS],
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_resp = await session.list_tools()
                    self._mcp_tools = tools_resp.tools
                    logger.info(f"[LLMClient] Discovered {len(self._mcp_tools)} tools via MCP.")
                    return self._mcp_tools
        except Exception as e:
            logger.error(f"[LLMClient] MCP Tool Discovery Failed: {e}")
            return []

    async def _execute_mcp_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Executes a specific tool via the MCP server.
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=settings.MCP_SERVER_COMMAND,
                args=[settings.MCP_SERVER_ARGS],
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
                    return result.content
        except Exception as e:
            logger.error(f"[LLMClient] MCP Tool Execution Failed ({name}): {e}")
            return f"Error executing tool: {str(e)}"

    async def generate_json(
        self,
        prompt: str,
        model_override: Optional[str] = None,
        retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Finalized generation with automated MCP tool calling.
        """
        model_name = model_override or self._default_model_name
        mcp_tools = await self._ensure_mcp_tools()
        
        # 1. Map MCP tools to Gemini FunctionDeclarations
        gemini_tools = []
        if mcp_tools:
            def clean_schema(s):
                if not isinstance(s, dict):
                    return s
                # Gemini prohibited fields
                forbidden = ["title", "additionalProperties", "default", "examples", "$schema"]
                for field in forbidden:
                    if field in s:
                        del s[field]
                if "properties" in s:
                    for k, v in s["properties"].items():
                        s["properties"][k] = clean_schema(v)
                return s

            declarations = []
            for tool in mcp_tools:
                params = clean_schema(tool.inputSchema.copy())
                
                declarations.append(genai.types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=params
                ))
            gemini_tools = [genai.types.Tool(function_declarations=declarations)]

        # 2. Get/Configure model with tools
        # We create a new model instance if tools are provided to ensure they are bound
        if gemini_tools:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={"response_mime_type": "application/json", "temperature": 0.2},
                tools=gemini_tools
            )
        else:
            model = self._get_model(model_name)
        
        loop = asyncio.get_event_loop()

        for attempt in range(retries):
            start = time.monotonic()
            try:
                # Use a chat session to handle multi-turn tool calling
                chat = model.start_chat(history=[], enable_automatic_function_calling=False)
                
                # First request
                response = await loop.run_in_executor(
                    _executor,
                    lambda: chat.send_message(prompt)
                )

                # 3. Tool Execution Loop
                # We handle one level of tool calling for now (standard for Phase 2)
                if response.candidates and response.candidates[0].content.parts:
                    content_parts = response.candidates[0].content.parts
                    if any(p.function_call for p in content_parts):
                        tool_responses = []
                        for part in content_parts:
                            if part.function_call:
                                call = part.function_call
                                logger.info(f"[LLMClient] Model requested tool: {call.name}")
                                
                                # Execute via MCP
                                tool_result = await self._execute_mcp_tool(call.name, dict(call.args))
                                
                                # Prepare function response
                                tool_responses.append(genai.types.Part(
                                    function_response=genai.types.FunctionResponse(
                                        name=call.name,
                                        response={"result": tool_result}
                                    )
                                ))
                        
                        # Feed back to model
                        response = await loop.run_in_executor(
                            _executor,
                            lambda: chat.send_message(tool_responses)
                        )

                # 4. Final Parse
                raw = response.text.strip()
                clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
                match = re.search(r"\{.*\}", clean, re.DOTALL)
                parsed = json.loads(match.group(0) if match else clean)

                latency_ms = int((time.monotonic() - start) * 1000)
                logger.debug(f"[LLMClient] generate_json (with tools) OK | latency={latency_ms}ms")
                return parsed

            except Exception as e:
                logger.warning(f"[LLMClient] Tool Loop Error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)
        
        return {"error": "LLM call with tools failed"}
    async def generate_text(
        self,
        prompt: str,
        model_override: Optional[str] = None,
    ) -> str:
        """
        Send a prompt and return raw text response.
        Useful for non-JSON use cases.
        """
        model = self._get_model(model_override)
        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(
                _executor,
                lambda: model.generate_content(prompt)
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"[LLMClient] generate_text failed: {type(e).__name__}: {e}")
            return ""

    def generate_json_sync(
        self,
        prompt: str,
        model_override: Optional[str] = None,
        timeout: int = 8,
    ) -> Dict[str, Any]:
        """
        Synchronous blocking version — used in ThreadPoolExecutor contexts
        (e.g., CoordinatorAgent._classify_intent runs in executor).

        Returns parsed dict or {"error": "..."}
        """
        model = self._get_model(model_override)
        start = time.monotonic()

        try:
            future = _executor.submit(model.generate_content, prompt)
            result_raw = future.result(timeout=timeout)
            raw = result_raw.text.strip()

            clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            parsed = json.loads(match.group(0) if match else clean)

            latency_ms = int((time.monotonic() - start) * 1000)
            logger.debug(
                f"[LLMClient] generate_json_sync OK | latency={latency_ms}ms"
            )
            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"[LLMClient] sync JSON parse error: {e}")
            return {"error": "JSON parse failed"}
        except Exception as e:
            logger.error(f"[LLMClient] sync call failed: {type(e).__name__}: {e}")
            return {"error": str(e)}


# ---------------------------------------------------------------------------
llm_client = LLMClient()
