"""
Exhaustive Alpha Vantage MCP Server
"""
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

from app.services.aidaan.mcp.shared import mcp


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_alphavantage_pro")


BASE_URL = "https://www.alphavantage.co/query"


def _is_simulatable_error(err: str) -> bool:
    low = (err or "").lower()
    return ("rate limit" in low) or ("not configured" in low)


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
                return {
                    "error": data["Error Message"]
                }
            if "Note" in data:
                return {
                    "error": "API rate limit reached (Notice from Alpha Vantage)"
                }
            return data
    except Exception as exc:
        logger.error(f"Alpha Vantage fetch failed: {exc}")
        return {"error": str(exc)}


# Core Market Data

@mcp.tool()
async def search_ticker(keywords: str) -> Dict[str, Any]:
    """
    Search for symbols or company names
    """
    data = await _fetch_av({
        "function": "SYMBOL_SEARCH",
        "keywords": keywords
    })

    if "error" in data:
        # Fallback to simulation if rate limited
        if (
            "rate limit" in data["error"].lower()
            or "not configured" in data["error"].lower()
        ):
            return {
                "matches": [
                    {
                        "symbol": keywords.upper(),
                        "name": f"{keywords.upper()} Corp (Simulated)",
                        "type": "Equity",
                        "region": "United States"
                    }
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
        if _is_simulatable_error(data["error"]):
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
async def get_daily_series(
    symbol: str,
    days: int = 10
) -> Dict[str, Any]:
    """
    Get daily historical stock prices (Open, High, Low, Close, Volume).
    Use 'days' to specify how many recent days of data to return.
    """
    raw = await _fetch_av(
        {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.upper(),
            "outputsize": "compact"
        }
    )

    if "error" in raw:
        if _is_simulatable_error(raw["error"]):
            import datetime
            today = datetime.date.today()
            sim_data = []
            for i in range(days):
                date_str = str(today - datetime.timedelta(days=i))
                sim_data.append({
                    "date": date_str,
                    "open": "900.00", "high": "910.00", "low": "890.00", "close": "905.00", "volume": "25000000"
                })
            return {
                "symbol": symbol.upper(),
                "data": sim_data,
                "note": "SIMULATION MODE ACTIVE"
            }
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
async def get_weekly_series(
    symbol: str,
    weeks: int = 5
) -> Dict[str, Any]:
    """
    Get weekly historical stock prices.
    Use 'weeks' to specify how many recent weeks of data to return.
    """
    raw = await _fetch_av({
        "function": "TIME_SERIES_WEEKLY",
        "symbol":symbol
    })

    if "error" in raw:
        if _is_simulatable_error(raw["error"]):
            return {
                "symbol": symbol.upper(), 
                "data": [], 
                "note": "SIMULATION MODE ACTIVE (History not available in simulation)"
            }
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


@mcp.tool()
async def get_intraday_series(
    symbol: str,
    interval: str = "5min",
    points: int = 50
) -> Dict[str, Any]:
    """
    Intraday OHLCV (e.g., 1min/5min/15min/30min/60min). Returns latest `points` rows.
    """
    interval = interval.strip()
    raw = await _fetch_av(
        {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol.upper(),
            "interval": interval,
            "outputsize": "compact"
        }
    )

    if "error" in raw:
        if _is_simulatable_error(raw["error"]):
            import datetime
            sim = []
            now = datetime.datetime.now()
            for i in range(max(1, min(points, 200))):
                ts = (now - datetime.timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S")
                sim.append({"ts": ts, "open": "900.00", "high": "910.00", "low": "890.00", "close": "905.00", "volume": "250000"})
            return {"symbol": symbol.upper(), "interval": interval, "data": sim, "note": "SIMULATION MODE ACTIVE"}
        return raw

    # Alpha Vantage key format: "Time Series (5min)"
    series_key = f"Time Series ({interval})"
    time_series = raw.get(series_key, {})

    if not time_series:
        return {
            "error": f"No intraday data found for interval={interval}"
        }
    
    sorted_ts = sorted(time_series.keys(), reverse=True)[: max(1, min(points, 200))]
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "last_refreshed": raw.get("Meta Data", {}).get("3. Last Refreshed"),
        "data": [
            {
                "ts": ts,
                "open": time_series[ts].get("1. open"),
                "high": time_series[ts].get("2. high"),
                "low": time_series[ts].get("3. low"),
                "close": time_series[ts].get("4. close"),
                "volume": time_series[ts].get("5. volume"),
            }
            for ts in sorted_ts
        ],
    }


@mcp.tool()
async def get_monthly_series(
    symbol: str, 
    months: int = 12
) -> Dict[str, Any]:
    """
    Monthly OHLCV. Returns latest `months` rows.
    """
    raw = await _fetch_av({
        "function": "TIME_SERIES_MONTHLY",
        "symbol": symbol.upper()
    })

    if "error" in raw:
        if _is_simulatable_error(raw["error"]):
            return {
                "symbol": symbol.upper(), 
                "data": [], 
                "note": "SIMULATION MODE ACTIVE (Monthly history not available in simulation)"
            }
        return raw

    time_series = raw.get("Monthly Time Series", {})
    if not time_series:
        return {
            "error": "No monthly data found."
        }
    
    sorted_dates = sorted(time_series.keys(), reverse=True)[: max(1, min(months, 120))]
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
                "volume": time_series[date].get("5. volume"),
            }
            for date in sorted_dates
        ]
    }


# Fundamental Data

@mcp.tool()
async def get_company_overview(symbol: str) -> Dict[str, Any]:
    """
    Get company fundamental (Sector, PE, Market Cap, etc)
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
    Includes revenue, gross profit, net income, etc...
    """
    return await _fetch_av({
        "function": "INCOME_STATEMENT",
        "symbol": symbol
    })


@mcp.tool()
async def get_balance_sheet(symbol: str) -> Dict[str, Any]:
    """
    get the annual and quarterly balance sheets for a company.
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
        return {
            "error": "No FX data found"
        }
    
    return {
        "from": rate.get("1. From_Currency Code"),
        "to": rate.get("3. To_Currency Code"),
        "rate": rate.get("5. Exchange Rate"),
        "last_refreshed": rate.get("6. Last Refreshed")
    }


@mcp.tool()
async def get_fx_daily_series(
    from_symbol: str,
    to_symbol: str,
    days: int = 30
) -> Dict[str, Any]:
    """
    FX daily OHLC (Time Series FX Daily). Returns latest `days` rows.
    """
    raw = await _fetch_av(
        {
            "function": "FX_DAILY",
            "from_symbol": from_symbol.upper(),
            "to_symbol": to_symbol.upper(),
            "outputsize": "compact",
        }
    )

    if "error" in raw:
        if _is_simulatable_error(raw["error"]):
            return {
                "pair": f"{from_symbol.upper()}/{to_symbol.upper()}", 
                "data": [], 
                "note": "SIMULATION MODE ACTIVE (FX history not available in simulation)"
            }
        return raw
    
    time_series = raw.get("Time Series FX (Daily)", {}) or raw.get("Time Series FX (Daily)", {})
    if not time_series:
        return {"error": "No FX daily data found."}

    sorted_dates = sorted(time_series.keys(), reverse=True)[: max(1, min(days, 200))]

    return {
        "pair": f"{from_symbol.upper()}/{to_symbol.upper()}",
        "last_refreshed": raw.get("Meta Data", {}).get("5. Last Refreshed") or raw.get("Meta Data", {}).get("3. Last Refreshed"),
        "data": [
            {
                "date": date,
                "open": time_series[date].get("1. open"),
                "high": time_series[date].get("2. high"),
                "low": time_series[date].get("3. low"),
                "close": time_series[date].get("4. close"),
            }
            for date in sorted_dates
        ],
    }


@mcp.tool()
async def get_fx_intraday_series(
    from_symbol: str,
    to_symbol: str,
    interval: str = "5min",
    points: int = 50
) -> Dict[str, Any]:
    """
    FX intraday OHLC (e.g., 5min/15min/30min/60min). Returns latest `points` rows.
    """
    interval = interval.strip()
    raw = await _fetch_av(
        {
            "function": "FX_INTRADAY",
            "from_symbol": from_symbol.upper(),
            "to_symbol": to_symbol.upper(),
            "interval": interval,
            "outputsize": "compact",
        }
    )

    if "error" in raw:
        if _is_simulatable_error(raw["error"]):
            return {
                "pair": f"{from_symbol.upper()}/{to_symbol.upper()}",
                "interval": interval,
                "data": [],
                "note": "SIMULATION MODE ACTIVE (FX intraday not available in simulation)"
            }
        return raw
    
    series_key = f"Time Series FX ({interval})"
    time_series = raw.get(series_key, {})

    if not time_series:
        return {
            "error": f"No FX intraday data found for interval={interval}"
        }
    
    sorted_ts = sorted(time_series.keys(), reverse=True)[: max(1, min(points, 200))]
    return {
        "pair": f"{from_symbol.upper()}/{to_symbol.upper()}",
        "interval": interval,
        "last_refreshed": raw.get("Meta Data", {}).get("5. Last Refreshed") or raw.get("Meta Data", {}).get("3. Last Refreshed"),
        "data": [
            {
                "ts": ts,
                "open": time_series[ts].get("1. open"),
                "high": time_series[ts].get("2. high"),
                "low": time_series[ts].get("3. low"),
                "close": time_series[ts].get("4. close"),
            }
            for ts in sorted_ts
        ]
    }


@mcp.tool()
async def get_crypto_daily_series(
    symbol: str,
    market: str = "USD",
    days: int = 30
) -> Dict[str, Any]:
    """
    Crypto daily OHLCV (Digital Currency Daily). Returns latest `days` rows.
    """
    raw = await _fetch_av(
        {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol.upper(),
            "market": market.upper()
        }
    )

    if "error" in raw:
        if _is_simulatable_error(raw["error"]):
            return {
                "symbol": symbol.upper(), 
                "market": market.upper(), 
                "data": [], 
                "note": "SIMULATION MODE ACTIVE (Crypto history not available in simulation)"
            }
        return raw

    time_series = raw.get("Time Series (Digital Currency Daily)", {})
    if not time_series:
        return {"error": "No crypto daily data found."}

    sorted_dates = sorted(time_series.keys(), reverse=True)[: max(1, min(days, 200))]

    # Values are usually duplicated in both base and quote currency; we return quote-currency keys when present.
    return {
        "symbol": symbol.upper(),
        "market": market.upper(),
        "last_refreshed": raw.get("Meta Data", {}).get("6. Last Refreshed") or raw.get("Meta Data", {}).get("5. Last Refreshed"),
        "data": [
            {
                "date": date,
                "open": time_series[date].get(f"1a. open ({market.upper()})") or time_series[date].get("1a. open (USD)"),
                "high": time_series[date].get(f"2a. high ({market.upper()})") or time_series[date].get("2a. high (USD)"),
                "low": time_series[date].get(f"3a. low ({market.upper()})") or time_series[date].get("3a. low (USD)"),
                "close": time_series[date].get(f"4a. close ({market.upper()})") or time_series[date].get("4a. close (USD)"),
                "volume": time_series[date].get("5. volume"),
                "market_cap": time_series[date].get("6. market cap (USD)"),
            }
            for date in sorted_dates
        ],
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
    valid = [
        "WTI",
        "BRENT",
        "NATURAL_GAS",
        "COPPER",
        "ALUMINUM",
        "WHEAT",
        "CORN",
        "COTTON",
        "SUGAR",
        "COFFEE",
        "ALL_COMMODITIES",
    ]
    if function not in valid:
        return {"error": f"Invalid function. Use one of: {valid}"}
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
    
    # Fetch data
    raw_data = await _fetch_av(params)
    
    if "error" in raw_data:
        return raw_data
        
    feed = raw_data.get("feed", [])
    if not feed:
        return raw_data
        
    # Analyze sentiment using locally deployed FinBERT Model
    from app.services.aidaan.core.sentiment import sentiment_analyzer
    
    # Extract titles for efficient batch processing
    titles = [item.get("title", "") for item in feed]
    analytics_status = "SUCCESS"
    
    try:
        # Run through our professional ML model
        sentiment_results = sentiment_analyzer.analyze_batch(titles)
        
        # Check if we fell back to neutral due to error
        if not sentiment_analyzer.is_loaded:
            analytics_status = "ERROR_FALLBACK"

        # Map intelligence results back to the news feed
        for item, sentiment in zip(feed, sentiment_results):
            # We inject the precise FinBERT scores required for the Visual State Machine
            item["finbert_sentiment_label"] = sentiment["label"]
            item["finbert_polarity_score"] = sentiment["score"] # Critical for "Alert/Busy" logic
            item["finbert_confidence"] = sentiment["confidence"]
    except Exception as e:
        import logging
        logging.getLogger("mcp_alphavantage_pro").error(f"FinBERT Analysis Failed: {e}")
        analytics_status = "ERROR_UNAVAILABLE"
        
    return {
        "feed": feed,
        "analytics_status": analytics_status,
        "note": "FinBERT Intelligence is currently unavailable" if analytics_status != "SUCCESS" else None
    }


# Technical Indicators
@mcp.tool()
async def get_technical_indicator(
    function: str,
    symbol: str,
    interval: str = "daily",
    time_period: Optional[int] = 14,
    series_type: str = "close",
    # Optional parameters for certain indicators (kept explicit to avoid schema confusion)
    fastperiod: Optional[int] = None,
    slowperiod: Optional[int] = None,
    signalperiod: Optional[int] = None,
    nbdevup: Optional[int] = None,
    nbdevdn: Optional[int] = None,
    matype: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generic technical indicator fetch (advanced).
    Supported functions (recommended): SMA, EMA, RSI, MACD, BBANDS.
    """
    valid = {"SMA", "EMA", "RSI", "MACD", "BBANDS"}
    fn = function.strip().upper()
    if fn not in valid:
        return {"error": f"Invalid function. Use one of: {sorted(valid)}"}

    params = {
        "function": fn,
        "symbol": symbol.upper(),
        "interval": interval,
    }
    if time_period:
        params["time_period"] = time_period
    params["series_type"] = series_type

    if fastperiod is not None:
        params["fastperiod"] = fastperiod
    if slowperiod is not None:
        params["slowperiod"] = slowperiod
    if signalperiod is not None:
        params["signalperiod"] = signalperiod
    if nbdevup is not None:
        params["nbdevup"] = nbdevup
    if nbdevdn is not None:
        params["nbdevdn"] = nbdevdn
    if matype is not None:
        params["matype"] = matype

    return await _fetch_av(params)


@mcp.tool()
async def get_sma(
    symbol: str,
    interval: str = "daily",
    time_period: int = 20,
    series_type: str = "close"
) -> Dict[str, Any]:
    """
    SMA for a symbol
    """
    return await _fetch_av(
        {
            "function": "SMA",
            "symbol": symbol.upper(),
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type,
        }
    )


@mcp.tool()
async def get_ema(
    symbol: str,
    interval: str = "daily",
    time_period: int = 20,
    series_type: str = "close"
) -> Dict[str, Any]:
    """
    EMA for a symbol.
    """
    return await _fetch_av(
        {
            "function": "EMA",
            "symbol": symbol.upper(),
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type
        }
    )


@mcp.tool()
async def get_rsi(
    symbol: str,
    interval: str = "daily",
    time_period: int = 14,
    series_type: str = "close"
) -> Dict[str, Any]:
    """
    RSI for a symbol
    """
    return await _fetch_av(
        {
            "function": "RSI",
            "symbol": symbol.upper(),
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type,
        }
    )


@mcp.tool()
async def get_macd(
    symbol: str,
    interval: str = "daily",
    series_type: str = "close",
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> Dict[str, Any]:
    """MACD for a symbol."""
    return await _fetch_av(
        {
            "function": "MACD",
            "symbol": symbol.upper(),
            "interval": interval,
            "series_type": series_type,
            "fastperiod": fastperiod,
            "slowperiod": slowperiod,
            "signalperiod": signalperiod,
        }
    )


@mcp.tool()
async def get_bbands(
    symbol: str,
    interval: str = "daily",
    time_period: int = 20,
    series_type: str = "close",
    nbdevup: int = 2,
    nbdevdn: int = 2,
    matype: int = 0,
) -> Dict[str, Any]:
    """Bollinger Bands (BBANDS) for a symbol."""
    return await _fetch_av(
        {
            "function": "BBANDS",
            "symbol": symbol.upper(),
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type,
            "nbdevup": nbdevup,
            "nbdevdn": nbdevdn,
            "matype": matype,
        }
    )


if __name__ == "__main__":
    mcp.run()
