"""
Alpha Vantage MCP Server
========================
Exposes trading and market data tools to AIDAAN via Model Context Protocol.
"""
import httpx
import logging
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_alphavantage")

# Initialize FastMCP server
mcp = FastMCP("Alpha Vantage Tool Server")

BASE_URL = "https://www.alphavantage.co/query"

# We will pass the API key via environment variable or context
# For now, we'll try to get it from a standard place or injected config
API_KEY = "PJBIMYS7Q8AYB7NJ" # Hardcoded as per collection, but should be dynamic

async def _fetch_av(params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to fetch from Alpha Vantage."""
    params["apikey"] = API_KEY
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

# --- TIME SERIES TOOLS ---

@mcp.tool()
async def get_stock_quote(symbol: str) -> Dict[str, Any]:
    """
    Get real-time stock quote for a given symbol.
    """
    logger.info(f"Fetching quote for {symbol}")
    return await _fetch_av({
        "function": "GLOBAL_QUOTE",
        "symbol": symbol
    })

@mcp.tool()
async def get_daily_series(symbol: str) -> Dict[str, Any]:
    """
    Get daily adjusted time series for a stock.
    """
    logger.info(f"Fetching daily series for {symbol}")
    return await _fetch_av({
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol
    })

# --- FUNDAMENTALS TOOLS ---

@mcp.tool()
async def get_company_overview(symbol: str) -> Dict[str, Any]:
    """
    Get fundamental company information like sector, PE ratio, dividend, etc.
    """
    logger.info(f"Fetching overview for {symbol}")
    return await _fetch_av({
        "function": "OVERVIEW",
        "symbol": symbol
    })

@mcp.tool()
async def get_income_statement(symbol: str) -> Dict[str, Any]:
    """
    Get annual and quarterly income statements for a company.
    """
    return await _fetch_av({
        "function": "INCOME_STATEMENT",
        "symbol": symbol
    })

# --- SENTIMENT TOOLS ---

@mcp.tool()
async def get_news_sentiment(tickers: str) -> Dict[str, Any]:
    """
    Get real-time news & sentiment for one or more tickers (comma-separated).
    """
    logger.info(f"Fetching news for {tickers}")
    return await _fetch_av({
        "function": "NEWS_SENTIMENT",
        "tickers": tickers
    })

# --- TECHNICAL INDICATORS ---

@mcp.tool()
async def get_rsi(symbol: str, interval: str = "daily", time_period: int = 14) -> Dict[str, Any]:
    """
    Get Relative Strength Index (RSI) for a symbol.
    """
    return await _fetch_av({
        "function": "RSI",
        "symbol": symbol,
        "interval": interval,
        "time_period": time_period,
        "series_type": "close"
    })

if __name__ == "__main__":
    mcp.run()
