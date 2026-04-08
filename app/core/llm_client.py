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

        logger.info(f"[LLMClient] Initialized. Default model: {self._default_model_name}")

        # Startup smoke test
        self._smoke_test()

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

    # ------------------------------------------------------------------
    async def generate_json(
        self,
        prompt: str,
        model_override: Optional[str] = None,
        retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Send a prompt and parse the response as JSON.

        Args:
            prompt:         The full prompt string (use Prompts.XYZ.format(...))
            model_override: Optional model name to use instead of the default
            retries:        How many times to retry on failure

        Returns:
            Parsed dict, or {"error": "..."} on failure

        Debug tip:
            Every call logs prompt hash + latency. Set LOG_LEVEL=DEBUG to see full prompts.
        """
        model = self._get_model(model_override)
        loop = asyncio.get_event_loop()

        for attempt in range(retries):
            start = time.monotonic()
            try:
                response = await loop.run_in_executor(
                    _executor,
                    lambda: model.generate_content(prompt)
                )
                raw = response.text.strip()

                # Strip markdown fences if present
                clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
                match = re.search(r"\{.*\}", clean, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                else:
                    parsed = json.loads(clean)

                latency_ms = int((time.monotonic() - start) * 1000)
                logger.debug(
                    f"[LLMClient] generate_json OK | model={model_override or self._default_model_name} "
                    f"| latency={latency_ms}ms | prompt_preview={prompt[:80]!r}"
                )
                return parsed

            except json.JSONDecodeError as e:
                logger.warning(
                    f"[LLMClient] JSON parse error (attempt {attempt + 1}/{retries}): {e} "
                    f"| raw={raw[:200]!r}"
                )
            except Exception as e:
                logger.warning(
                    f"[LLMClient] API error (attempt {attempt + 1}/{retries}): "
                    f"{type(e).__name__}: {e}"
                )

            if attempt < retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))  # simple backoff

        logger.error(
            f"[LLMClient] All {retries} attempts failed. "
            f"Prompt preview: {prompt[:200]!r}"
        )
        return {"error": f"LLM call failed after {retries} attempts"}

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
