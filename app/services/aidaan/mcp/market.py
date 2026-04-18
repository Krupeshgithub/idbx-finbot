"""
Exhaustive Alpha Vantage MCP Server
"""
import logging
import os
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
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {
            "error": "ALPHA_VANTAGE_API_KEY is not configured."
        }

    request_params = dict(params)
    request_params["apikey"] = api_key

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
    
    if "error" in data:
        # Fallback to simulation if rate limited
        if "rate limit" in data["error"].lower() or "not configured" in data["error"].lower():
            return {
                "matches": [
                    {"symbol": keywords.upper(), "name": f"{keywords.upper()} Corp (Simulated)", "type": "Equity", "region": "United States"}
                ],
                "note": "SIMULATION MODE ACTIVE"
            }
        return data
        
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
    data = await _fetch_av({
        "function": "GLOBAL_QUOTE",
        "symbol": symbol.upper()
    })
    
    if "error" in data:
        if "rate limit" in data["error"].lower() or "not configured" in data["error"].lower():
            import random
            return {
                "symbol": symbol.upper(),
                "price": str(round(random.uniform(100, 1000), 2)),
                "change_percent": f"{round(random.uniform(-5, 5), 2)}%",
                "volume": str(random.randint(1000000, 50000000)),
                "latest_day": "2026-04-16 (Simulated)",
                "note": "SIMULATION MODE ACTIVE"
            }
        return data

    quote = data.get("Global Quote", {})
    if not quote: return {"error": "No data found"}

    return {
        "symbol": quote.get("01. symbol"),
        "price": quote.get("05. price"),
        "change_percent": quote.get("10. change percent"),
        "volume": quote.get("06. volume"),
        "latest_day": quote.get("07. latest trading day")
    }


@mcp.tool()
async def get_daily_series(symbol: str, days: int = 10) -> Dict[str, Any]:
    """
    Get daily historical stock prices (Open, High, Low, Close, Volume).
    Use 'days' to specify how many recent days of data to return.
    """
    raw = await _fetch_av(
        {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.upper(),
            "outputsize": "compact",
        }
    )

    if "error" in raw:
        if "rate limit" in raw["error"].lower() or "not configured" in raw["error"].lower():
            import datetime
            today = datetime.date.today()
            sim_data = []
            for i in range(days):
                date_str = str(today - datetime.timedelta(days=i))
                sim_data.append({
                    "date": date_str,
                    "open": "900.00", "high": "910.00", "low": "890.00", "close": "905.00", "volume": "25000000"
                })
            return {"symbol": symbol.upper(), "data": sim_data, "note": "SIMULATION MODE ACTIVE"}
        return raw
    
    time_series = raw.get("Time Series (Daily)", {})
    if not time_series:
        return {"error": "No daily data found."}
    
    # Slicing the last X days for efficiency
    sorted_dates = sorted(time_series.keys(), reverse=True)
    target_dates = sorted_dates[:days]
    
    return {
        "symbol": symbol.upper(),
        "last_refreshed": raw.get("Meta Data", {}).get("3. Last Refreshed"),
        "data": [
            {
                "date": date,
                "open": time_series[date].get("1. open"),
                "high": time_series[date].get("2. high"),
                "low": time_series[date].get("3. low"),
                "close": time_series[date].get("4. close"),
                "volume": time_series[date].get("5. volume")
            } for date in target_dates
        ]
    }


@mcp.tool()
async def get_weekly_series(symbol: str, weeks: int = 5) -> Dict[str, Any]:
    """
    Get weekly historical stock prices.
    Use 'weeks' to specify how many recent weeks of data to return.
    """
    raw = await _fetch_av({
        "function": "TIME_SERIES_WEEKLY",
        "symbol": symbol
    })
    
    if "error" in raw:
        if "rate limit" in raw["error"].lower():
            return {"symbol": symbol.upper(), "data": [], "note": "SIMULATION MODE ACTIVE (History not available in simulation)"}
        return raw
    
    time_series = raw.get("Weekly Time Series", {})
        
    sorted_dates = sorted(time_series.keys(), reverse=True)
    target_dates = sorted_dates[:weeks]
    
    return {
        "symbol": symbol,
        "data": [
            {
                "date": date,
                "open": time_series[date].get("1. open"),
                "high": time_series[date].get("2. high"),
                "low": time_series[date].get("3. low"),
                "close": time_series[date].get("4. close"),
                "volume": time_series[date].get("5. volume")
            } for date in target_dates
        ]
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


@mcp.tool()
async def get_income_statement(symbol: str) -> Dict[str, Any]:
    """
    Get the annual and quarterly income statements for a company.
    Includes revenue, gross profit, net income, etc.
    """
    return await _fetch_av({
        "function": "INCOME_STATEMENT",
        "symbol": symbol
    })


@mcp.tool()
async def get_balance_sheet(symbol: str) -> Dict[str, Any]:
    """
    Get the annual and quarterly balance sheets for a company.
    Includes assets, liabilities, and equity.
    """
    return await _fetch_av({
        "function": "BALANCE_SHEET",
        "symbol": symbol
    })


@mcp.tool()
async def get_cash_flow(symbol: str) -> Dict[str, Any]:
    """
    Get the annual and quarterly cash flow statements for a company.
    Includes operating, investing, and financing cash flows.
    """
    return await _fetch_av({
        "function": "CASH_FLOW",
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
    if "error" in data:
        return data

    rate = data.get("Realtime Currency Exchange Rate", {})
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
    time_period: Optional[int] = 14,
    **kwargs
) -> Dict[str, Any]:
    """
    Get professional technical indicators from Alpha Vantage.
    
    Supported functions: SMA, EMA, WMA, RSI, MACD, STOCH, BBANDS, ADX, etc.
    Advanced params (passed via kwargs):
    - MACD: fastperiod=12, slowperiod=26, signalperiod=9
    - BBANDS: nbdevup=2, nbdevdn=2, matype=0
    - Performance: series_type=close (standard)
    """
    params = {
        "function": function,
        "symbol": symbol.upper(),
        "interval": interval,
    }
    if time_period:
        params["time_period"] = time_period
    
    if "series_type" not in kwargs:
        params["series_type"] = "close"
        
    params.update(kwargs)
    return await _fetch_av(params)

# Gainers and Losers tools

@mcp.tool()
async def get_top_gainers_losers() -> Dict[str, Any]:
    """
    Get today's top 20 gainers, losers, and most actively traded tickers.
    Very fast endpoint to understand overall market sentiment at a glance.
    """
    return await _fetch_av({
        "function": "TOP_GAINERS_LOSERS"
    })


# Intraday series tools

@mcp.tool()
async def get_intraday_series(symbol: str, interval: str = "5min") -> Dict[str, Any]:
    """
    Get live intraday stock prices.
    Intervals: 1min, 5min, 15min, 30min, 60min.
    """
    return await _fetch_av({
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol.upper(),
        "interval": interval,
    })


# Crypto rating tools

@mcp.tool()
async def get_crypto_rating(symbol: str) -> Dict[str, Any]:
    """
    Get the FCAS rating (Fundamental Crypto Asset Score) for a Crypto asset.
    """
    return await _fetch_av({
        "function": "CRYPTO_RATING",
        "symbol": symbol.upper()
    })


# Insider transactions tools

@mcp.tool()
async def get_insider_transactions(symbol: str) -> Dict[str, Any]:
    """
    Get the latest insider trading transactions (officers, directors) for a company.
    Helps detect if executives are buying or selling their own company's stock.
    """
    return await _fetch_av({
        "function": "INSIDER_TRANSACTIONS",
        "symbol": symbol.upper()
    })


# --- Deeply Sliced (Fast Performance) Tools ---

@mcp.tool()
async def get_key_company_metrics(symbol: str) -> Dict[str, Any]:
    """
    Get ONLY the most critical metrics (PE, Market Cap, Dividend, EPS, 52WeekHigh/Low).
    Divided from 'get_company_overview' to return a tiny, ultra-fast JSON payload.
    """
    data = await _fetch_av({
        "function": "OVERVIEW",
        "symbol": symbol.upper()
    })
    
    if "error" in data:
        return data
        
    return {
        "Symbol": data.get("Symbol"),
        "Name": data.get("Name"),
        "Sector": data.get("Sector"),
        "MarketCapitalization": data.get("MarketCapitalization"),
        "PERatio": data.get("PERatio"),
        "DividendYield": data.get("DividendYield"),
        "EPS": data.get("EPS"),
        "52WeekHigh": data.get("52WeekHigh"),
        "52WeekLow": data.get("52WeekLow"),
    }


@mcp.tool()
async def get_latest_financial_report(symbol: str, report_type: str = "INCOME_STATEMENT") -> Dict[str, Any]:
    """
    Get ONLY the most recent Annual and Quarterly financial report (Income, Balance, or CashFlow).
    Use this instead of the full historical tools when the user only asks for "current/latest" data.
    Valid report_type: INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW
    """
    data = await _fetch_av({
        "function": report_type.upper(),
        "symbol": symbol.upper()
    })
    
    if "error" in data:
        return data
        
    # SLICING: Get only the [0]th index (the latest report) instead of 5 years of data
    latest_annual = data.get("annualReports", [{}])[0] if data.get("annualReports") else {}
    latest_quarterly = data.get("quarterlyReports", [{}])[0] if data.get("quarterlyReports") else {}
    
    return {
        "symbol": data.get("symbol"),
        "report_type": report_type,
        "latest_annual_report": latest_annual,
        "latest_quarterly_report": latest_quarterly
    }


if __name__ == "__main__":
    mcp.run()
