"""
Shared trade-parsing helpers for AIDAAN agents.
"""
from __future__ import annotations

import re
from typing import Optional


_INSTRUMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("eur/usd", "EUR/USD"),
    ("gbp/usd", "GBP/USD"),
    ("cable", "GBP/USD"),
    ("sonia", "SONIA"),
    ("sofr", "SOFR"),
    ("uk gilts", "UK GILTS"),
    ("gilt", "UK GILTS"),
    ("bond", "BOND"),
    ("fx", "FX"),
)

# Common stock ticker mappings for quick resolution
_STOCK_TICKER_ALIASES: dict[str, str] = {
    "google": "GOOGL",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "intel": "INTC",
    "amd": "AMD",
    "ibm": "IBM",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "netflix": "NFLX",
    "uber": "UBER",
    "airbnb": "ABNB",
    "spotify": "SPOT",
    "zoom": "ZM",
    "slack": "SLACK",
    "shopify": "SHOP",
    "stripe": "STRIPE",
    "paypal": "PYPL",
    "square": "SQ",
    "coinbase": "COIN",
    "robinhood": "HOOD",
    "twitter": "TWTR",
    "x": "TWTR",
    "reddit": "RDDT",
    "discord": "DISCORD",
    "tiktok": "TIKTOK",
    "snapchat": "SNAP",
    "pinterest": "PINS",
    "lyft": "LYFT",
    "doordash": "DASH",
    "airbnb": "ABNB",
    "booking": "BKNG",
    "expedia": "EXPE",
    "tripadvisor": "TRIP",
    "yelp": "YELP",
    "zillow": "Z",
    "redfin": "RDFN",
    "trulia": "TRULIA",
}


def parse_notional(text: str) -> Optional[float]:
    """
    Parse a human-readable notional such as ``25m`` or ``₹1.5 Crores`` into base units.
    """
    if not text:
        return None
    # Remove all commas for safety before parsing
    clean_text = text.replace(",", "").lower()
    
    # Require a magnitude character/word so we don't accidentally parse "7:30 AM" or "12 portfolios"
    matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*(k|lakh|lakhs|cr|crore|crores|bn|b|m|mm|million|billion)\b", clean_text))
    
    if not matches:
        return None

    # Generally take the last mentioned magnitude in the sentence if multiple exist (assumes the trailing intent)
    match = matches[-1]

    value = float(match.group(1))
    unit = (match.group(2) or "").lower()
    
    if unit in {"k"}:
        return value * 1_000
    if unit in {"lakh", "lakhs"}:
        return value * 100_000
    if unit in {"m", "mm", "million"}:
        return value * 1_000_000
    if unit in {"cr", "crore", "crores"}:
        return value * 10_000_000
    if unit in {"bn", "b", "billion"}:
        return value * 1_000_000_000
    return value


def parse_tenor(text: str) -> Optional[str]:
    """
    Parse tenor strings such as ``5Y`` or ``3 months`` into canonical format.
    """
    match = re.search(r"\b(\d+)\s*(y|yr|year|years|m|mo|month|months)\b", text.lower())
    if not match:
        match = re.search(r"\b(\d+)(y|m)\b", text.lower())
    if not match:
        return None

    qty = match.group(1)
    unit = match.group(2).lower()
    if unit in {"y", "yr", "year", "years"}:
        return f"{qty}Y"
    return f"{qty}M"


def parse_settlement(text: str) -> Optional[str]:
    """
    Parse common settlement conventions from free text.
    """
    lowered = text.lower()
    if "imm" in lowered:
        return "IMM"
    if "t+2" in lowered:
        return "T+2"
    if "spot" in lowered:
        return "Spot"
    return None


def infer_instrument(text: str, *, default: str = "DEFAULT") -> str:
    """
    Infer the primary instrument or asset label from user text.
    """
    lowered = text.lower()
    for pattern, instrument in _INSTRUMENT_PATTERNS:
        if pattern in lowered:
            return instrument
    return default


def resolve_stock_ticker(text: str) -> Optional[str]:
    """
    Resolve common company names to their stock tickers.
    Returns the ticker symbol if found, None otherwise.
    
    Example:
        resolve_stock_ticker("google price") -> "GOOGL"
        resolve_stock_ticker("apple stock") -> "AAPL"
    """
    lowered = text.lower()
    
    # Check for exact word matches in the text
    for company_name, ticker in _STOCK_TICKER_ALIASES.items():
        # Use word boundary to avoid partial matches
        if re.search(rf"\b{re.escape(company_name)}\b", lowered):
            return ticker
    
    return None

