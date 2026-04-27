"""
Unified MCP entrypoint for market and operational tools.
"""
from app.services.aidaan.mcp.shared import mcp

# Register tool modules on import.
from app.services.aidaan.mcp import market as _market_tools  # noqa: F401
from app.services.aidaan.mcp import operational_tools as _operational_tools  # noqa: F401
from app.services.aidaan.mcp import a2a_tools as _a2a_tools  # noqa: F401

if __name__ == "__main__":
    mcp.run()
