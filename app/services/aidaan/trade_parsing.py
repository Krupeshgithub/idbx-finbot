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


def parse_notional(text: str) -> Optional[float]:
    """
    Parse a human-readable notional such as ``25m`` or ``1.5bn`` into base units.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*(bn|b|m|mm|million|billion)?\b", text.lower())
    if not match:
        return None

    value = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if unit in {"bn", "b", "billion"}:
        return value * 1_000_000_000
    if unit in {"m", "mm", "million"}:
        return value * 1_000_000
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
