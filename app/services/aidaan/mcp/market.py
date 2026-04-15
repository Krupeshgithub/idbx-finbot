"""
Exhaustive Alpha Vantage MCP Server
"""
import logging
from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_alphavantage_pro")

mcp = FastMCP("Alpha Vantage Professional Server")


BASE_URL = "https://www.alphavantage.co/query"


async def _fetch_av(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    HTTP helper for Alpha Vantage.
    """
    _key = "PJBIMYS7Q8AYB7NJ"

    request_params = dict(params)
    request_params["apikey"] = _key

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(BASE_URL, params=request_params)
            response.raise_for_status()
            data = response.json()

            # Check for API-level errors
            if "Error Message" in data:
                return {"error": data["Error Message"]}
            if "Note" in data:
                return {"error": "API rate limit reached (Notice from Alpha Vantage)"}
        
            return data
    except Exception as exc:
        logger.error(f"Alpha Vantage fetch failed: {exc}")
        return {"error": str(exc)}
    

# Core Market Data

@mcp.tool()
async def search_ticker(keywords: str) -> Dict[str, Any]:
    """
    Search for symbols or company names.
    """
    data = await _fetch_av({
        "function": "SYMBOL_SEARCH",
        "keywords": keywords
    })
    matches = data.get("bestMatches", [])
    return {
        "matches": [
            {
                "symbol": m.get("1. symbol"), 
                "name": m.get("2. name"), 
                "type": m.get("3. type"), 
                "region": m.get("4. region")
            } for m in matches[:5]
        ]
    }


@mcp.tool()
async def get_stock_quote(symbol: str) -> Dict[str, Any]:
    """
    Get real-time quote for a symbol.
    """
    raw = await _fetch_av({
        "function": "GLOBAL_QUOTE",
        "symbol": symbol
    })
    quote = raw.get("Global Quote", {})
    if not quote: return {"error": "No data found"}

    return {
        "symbol": quote.get("01. symbol"),
        "price": quote.get("05. price"),
        "change_percent": quote.get("10. change percent"),
        "volume": quote.get("06. volume"),
        "latest_day": quote.get("07. latest trading day")
    }


# Fundamental Data

@mcp.tool()
async def get_company_overview(symbol: str) -> Dict[str, Any]:
    """
    Get company fundamental (Sector, PE, Market Cap, etc).
    """
    return await _fetch_av({
        "function": "OVERVIEW",
        "symbol": symbol
    })


@mcp.tool()
async def get_earnings(symbol: str) -> Dict[str, Any]:
    """
    Get historical and projected earnings (EPS)
    """
    return await _fetch_av({
        "function": "EARNINGS",
        "symbol": symbol
    })


# Forex (FX)

@mcp.tool()
async def get_exchange_rate(
    from_currency: str,
    to_currency: str
) -> Dict[str, Any]:
    """
    Get real-time exchange rate between two currencies (e.g. GBP to USD)
    """
    data = await _fetch_av({
        "function": "CURRENCY_EXCHANGE_RATE", 
        "from_currency": from_currency, 
        "to_currency": to_currency
    })
    rate = data.get("Raltime Currency Exchange Rate", {})
    if not rate:
        return {"error": "No FX data found"}

    return {
        "from": rate.get("1. From_Currency Code"),
        "to": rate.get("3. To_Currency Code"),
        "rate": rate.get("5. Exchange Rate"),
        "last_refreshed": rate.get("6. Last Refreshed")
    }


# Economic Indicators (High Interest for Institutions)

@mcp.tool()
async def get_economic_indicator(function: str) -> Dict[str, Any]:
    """
    Get key economic indicators.
    Supported functions: REAL_GDP, CPI, INFLATION, UNEMPLOYMENT, FEDERAL_FUNDS_RATE.
    """
    valid = [
        "REAL_GDP",
        "CPI",
        "INFLATION",
        "UNEMPLOYMENT",
        "FEDERAL_FUNDS_RATE"
    ]
    if function not in valid:
        return {
            "error": f"Invalid function. Use one of: {valid}"
        }
    return await _fetch_av({"function": function})


# Commodities

@mcp.tool()
async def get_commodity_price(function: str) -> Dict[str, Any]:
    """
    Get historical prices for commodities.
    Supported functions:  WTI, BRENT, NATURAL_GAS, COPPER, ALUMINUM, WHEAT, CORN, COTTON, SUGAR, COFFEE, ALL_COMMODITIES.
    """
    return await _fetch_av({"function": function})


# Alpha Intelligence (News & Sentiment)

@mcp.tool()
async def get_market_news(
    tickers: Optional[str] = None,
    topics: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get real-time market news and sentiment.
    Tickers can be comma-separated.
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "limit": 10
    }
    if tickers: params["tickers"] = tickers
    if topics: params["topics"] = topics
    return await _fetch_av(params)


# Technical Indicators

@mcp.tool()
async def get_technical_indicator(
    function: str,
    symbol: str,
    interval: str = "daily",
    time_period: int = 14
) -> Dict[str, Any]:
    """
    Get technical indicators (RSI, SMA, EMA, MACD, etc)
    """
    params = {
        "function": function,
        "symbol": symbol,
        "interval": interval,
        "time_period": time_period,
        "series_type": "close"
    }
    return await _fetch_av(params)


if __name__ == "__main__":
    mcp.run()
