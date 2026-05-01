"""
Agent-to-Agent (A2A) MCP Tools
"""
import logging
import time
from typing import Any, Dict

from app.services.aidaan.mcp.shared import mcp
from app.services.aidaan.core.base import registry
from app.services.aidaan.runtime_context import current_conversation_id, current_username

logger = logging.getLogger("mcp_a2a_tools")


@mcp.tool()
async def consult_specialist_agent(
    target_agent: str,
    query: str,
) -> Dict[str, Any]:
    """
    Agent-to-Agent (A2A) Protocol Tool.
    Use this to securely query another specialized agent (e.g. 'risk', 'context', 'order') within the Zero Leakage Privacy Vault.
    Provide the target_agent name and your explicit query text.
    """
    start_time = time.monotonic()
    
    # Extract session boundary from securely propagated contextvars
    conv_id = current_conversation_id.get("")
    uname = current_username.get(None)

    logger.info(
        "[A2A_SPINE] >>> INITIATING SECURE A2A HANDSHAKE >>> | target=%s | conv_id=%s | user=%s",
        target_agent, conv_id, uname
    )
    logger.info("[A2A_QUERY] query_len=%s | preview=%s", len(query), query[:200])
    
    agent = registry.get_agent(target_agent)
    if not agent:
        logger.error("[A2A_ERROR] ❌ Agent '%s' not found in registry. Available: %s",
                     target_agent, registry.list_agents())
        return {"error": f"Agent '{target_agent}' not found in registry."}

    if not conv_id:
        logger.error("[A2A_ERROR] ❌ Missing conversation_id — rejecting to prevent context leakage.")
        return {"error": "Secure execution failed. Missing conversation boundary context."}

    try:
        logger.info("[A2A_EXECUTE] 🏃 Delegating to %s.handle_message() | conv_id=%s", target_agent, conv_id)
        response = await agent.handle_message(
            text=query,
            conversation_id=conv_id,
            context={"username": uname, "is_a2a_call": True}
        )
        
        latency = int((time.monotonic() - start_time) * 1000)
        reply_preview = response.reply[:200] + "..." if len(response.reply) > 200 else response.reply
        logger.info(
            "[A2A_SPINE] <<< A2A HANDSHAKE SUCCESS <<< | target=%s | latency_ms=%s | bullets=%s",
            target_agent, latency, len(response.bullets)
        )
        logger.info("[A2A_RESPONSE] %s", reply_preview)
        
        # We must return a plain dictionary for the LLM tool parsing
        return {
            "status": "success",
            "agent_reply": response.reply,
            "bullets": response.bullets,
            "actions_proposed": [act.get("type") for act in response.actions] if response.actions else [],
            "latency_ms": response.latency_ms,
            "system_latency_ms": latency
        }
    except Exception as exc:
        latency = int((time.monotonic() - start_time) * 1000)
        logger.error(
            "[A2A_ERROR] ❌ Call to '%s' FAILED | latency_ms=%s | error=%s",
            target_agent, latency, exc
        )
        return {"error": f"A2A call failed: {exc}"}
