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


@router.get("/diagnostics/vertex-ai-integration", tags=["diagnostics"])
async def diagnostics_vertex_ai_integration():
    """
    Check Vertex AI Cloud SQL integration status.
    
    Returns comprehensive diagnostics including:
    - Extension installation status
    - Embedding function availability
    - Vector column existence
    - Embedding generation test
    - Coverage statistics
    - Recommendations for setup
    """
    from app.db.session import get_db_session
    from app.db.vertex_ai_utils import VertexAIIntegrationStatus
    
    with get_db_session() as session:
        status = VertexAIIntegrationStatus.get_integration_status(session)
        
        return {
            "integration_ready": status["integration_ready"],
            "checks": {
                "extension_installed": status["extension_installed"],
                "embedding_function_exists": status["embedding_function_exists"],
                "vector_column_exists": status["vector_column_exists"],
                "embedding_test_passed": status["embedding_test_passed"],
            },
            "performance": {
                "embedding_test_latency_ms": status.get("embedding_test_latency_ms"),
                "status": (
                    "excellent" if status.get("embedding_test_latency_ms", 999) < 50
                    else "good" if status.get("embedding_test_latency_ms", 999) < 100
                    else "acceptable" if status.get("embedding_test_latency_ms", 999) < 200
                    else "slow" if status.get("embedding_test_latency_ms") is not None
                    else "unavailable"
                )
            },
            "coverage": {
                "total_messages": status.get("total", 0),
                "with_embeddings": status.get("with_embeddings", 0),
                "without_embeddings": status.get("without_embeddings", 0),
                "coverage_percentage": status.get("coverage_pct", 0.0),
                "status": (
                    "excellent" if status.get("coverage_pct", 0) >= 95
                    else "good" if status.get("coverage_pct", 0) >= 80
                    else "fair" if status.get("coverage_pct", 0) >= 50
                    else "poor"
                )
            },
            "recommendations": status.get("recommendations", []),
            "configuration": {
                "semantic_limit": settings.AIDAAN_SEMANTIC_LIMIT,
                "semantic_threshold": settings.AIDAAN_SEMANTIC_THRESHOLD,
                "history_window": settings.AIDAAN_HISTORY_WINDOW,
                "cache_ttl_seconds": settings.CONTEXT_CACHE_TTL_SECONDS,
                "cache_max_size": settings.CACHE_MAX_SIZE,
            }
        }
