"""
Operational diagnostics routes for AIDAAN.
Safe, non-sensitive introspection for local/dev verification.
"""

from fastapi import APIRouter

from app.core.cache import cache
from app.core.llm_client import llm_client
from app.core.config.settings import settings
from app.services.kill_switch import kill_switch_service


router = APIRouter()


@router.get("/diagnostics/config", tags=["diagnostics"])
async def diagnostics_config():
    # Vertex status
    vertex_ok = llm_client._ensure_client() is not None  # noqa: SLF001
    vertex_mode = getattr(llm_client, "_runtime_mode", "unknown")  # noqa: SLF001
    vertex_reason = getattr(llm_client, "_vertex_disabled_reason", None)  # noqa: SLF001

    # MCP status
    tools = await llm_client._ensure_mcp_tools()  # noqa: SLF001
    tool_names = [t.name for t in tools][:25]

    # Cache backend signal (fakeredis vs redis)
    cache_backend = cache.client.__class__.__name__

    return {
        "vertex": {
            "enabled": vertex_ok,
            "mode": vertex_mode,
            "disabled_reason": vertex_reason,
            "project": settings.GOOGLE_CLOUD_PROJECT,
            "location": settings.GOOGLE_CLOUD_LOCATION,
            "api_version": settings.VERTEX_AI_API_VERSION,
            "default_model": settings.VERTEX_AI_MODEL_NAME,
            "reasoning_model": settings.VERTEX_AI_REASONING_MODEL_NAME,
        },
        "mcp": {
            "transport": settings.MCP_TRANSPORT,
            "server_command": settings.MCP_SERVER_COMMAND,
            "server_args": settings.MCP_SERVER_ARGS,
            "tool_count": len(tools),
            "tool_names_sample": tool_names,
        },
        "cache": {
            "backend": cache_backend,
            "redis_url": settings.REDIS_URL,
        },
        "kill_switch": kill_switch_service.get_status(),
    }

